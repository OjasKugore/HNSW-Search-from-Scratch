# HNSW — Data Structures & Methodology

> A deep-dive into the exact data structures and high-level algorithm used in this C implementation.

---

## Table of Contents

1. [Core Intuition](#core-intuition)
2. [Data Structures](#data-structures)
   - [Vector](#1-vector)
   - [HeapItem & Heap](#2-heapitem--heap)
   - [VisitedSet](#3-visitedset)
   - [HNSWNode](#4-hnswnode)
   - [HNSWIndex](#5-hnswindex)
   - [KNNResult](#6-knnresult)
3. [High-Level Methodology](#high-level-methodology)
   - [Layer Structure](#layer-structure)
   - [Inserting a Vector](#inserting-a-vector)
   - [Searching for Nearest Neighbours](#searching-for-nearest-neighbours)
   - [The Core: search_layer()](#the-core-search_layer)
   - [Neighbour Selection](#neighbour-selection)
4. [How the Data Structures Connect](#how-the-data-structures-connect)
5. [Complexity Summary](#complexity-summary)

---

## Core Intuition

HNSW organises all vectors into a **multi-layer proximity graph**. Think of it like a highway system:

```
Layer 2:  A ─────────────────── B        <- Motorway (few nodes, long jumps)
           \                   /
Layer 1:  A ─ C ─ D ───────── B ─ E      <- Main road (medium density)
           \  |  \           / \
Layer 0:  A-C-D-F-G-H-I-J-K-B-E-...     <- Street level (ALL vectors, most edges)
```

- Upper layers act as **express lanes** — you jump large distances quickly.
- Lower layers allow **fine-grained local search**.
- Every vector lives at layer 0. Only a random subset reach higher layers.

The result: nearest neighbour search runs in **O(log N)** instead of brute-force O(N x dim).

---

## Data Structures

### 1. `Vector`

**File:** `src/vector.h`, `src/vector.c`

The fundamental unit of data. Wraps a heap-allocated float array with metadata.

```c
typedef struct {
    float  *data;    // The actual coordinate array, e.g. float[128]
    int     dim;     // Dimensionality (e.g. 128, 384, 768)
    int     id;      // Unique integer identifier
    char   *label;   // Optional string label (e.g. "cat.jpg", "king")
} Vector;
```

**Memory layout:**
```
Vector (on heap)
+-- id     = 42
+-- dim    = 4
+-- label  = "apple"
+-- data ---> [ 0.12 | 0.87 | -0.43 | 0.55 ]   (float array on heap)
```

**Distance functions** live here too:
```c
float dist_l2_sq(const float *a, const float *b, int dim);         // ||a - b||^2
float dist_cosine(const float *a, const float *b, int dim);        // 1 - cos(a,b)
float dist_inner_product(const float *a, const float *b, int dim); // -a.b
```

> Squared Euclidean is preferred because it avoids the `sqrt()` call and preserves ordering.

---

### 2. `HeapItem` & `Heap`

**File:** `src/heap.h`, `src/heap.c`

The **most-used data structure in the entire project**. Called on every `search_layer` invocation (which happens during both insert and search).

```c
typedef struct {
    int   node_id;   // Which graph node this refers to
    float distance;  // Distance from the query vector to this node
} HeapItem;

typedef struct {
    HeapItem *items;    // Contiguous array of HeapItems (the heap array)
    int       size;     // Current number of elements
    int       capacity; // Allocated capacity (doubles when full)
    int       type;     // HEAP_MIN (0) or HEAP_MAX (1)
} Heap;
```

**Internal array layout (for a min-heap of size 5):**

```
Index:    0        1        2        3        4
       [0.1, 3] [0.4, 7] [0.3, 2] [0.9, 1] [0.7, 5]
         ^ root (smallest distance)

Parent of index i  ->  (i-1) / 2
Left child of i    ->  2*i + 1
Right child of i   ->  2*i + 2
```

**Why you need BOTH min and max heaps:**

| Heap Type | Role in HNSW | Why |
|-----------|-------------|-----|
| **Min-Heap** (`candidates`) | Candidate queue during `search_layer` | Always expand the *closest* unvisited node next |
| **Max-Heap** (`results`) | Result buffer during `search_layer` | The root = farthest known result; prune it when buffer exceeds `ef` |

This combination is what makes the beam-search pruning condition efficient — checking if further exploration can improve results is just a `heap_peek()` on the max-heap root.

---

### 3. `VisitedSet`

**File:** `src/utils.h`, `src/utils.c`

Tracks which nodes have been visited during a single graph traversal. A key performance trick: **resetting is O(1)** using a generation counter.

```c
typedef struct {
    int *gen;         // gen[node_id] = generation when this node was last visited
    int  current_gen; // The current query generation number
    int  capacity;    // Total number of nodes tracked
} VisitedSet;
```

**Why not just use `bool visited[N]`?**

Clearing a `bool` array is **O(N)** per query — catastrophic for 100K+ nodes. The generation trick avoids this:

```
Query 1: current_gen = 1
  -> Mark node 5: gen[5] = 1
  -> Mark node 9: gen[9] = 1

Query 2: visited_reset() -> current_gen = 2   <- O(1)!
  -> Node 5 is NOT visited (gen[5] == 1 != 2)
  -> Mark node 5: gen[5] = 2
```

---

### 4. `HNSWNode`

**File:** `src/graph.h`, `src/graph.c`

A single node in the multi-layer graph. The critical design: **each layer has its own independent adjacency list**.

```c
typedef struct {
    int   id;                  // Same as its index in HNSWIndex.nodes[]
    int   level;               // Highest layer this node appears in (0-indexed)
    int **neighbors;           // neighbors[layer] -> int array of neighbor IDs
    int  *neighbor_count;      // Number of valid entries in neighbors[layer]
    int  *neighbor_capacity;   // Allocated capacity of neighbors[layer]
} HNSWNode;
```

**Memory layout for a node at level 2:**

```
HNSWNode (id=7, level=2)
+-- neighbors[0] -> [ 3 | 11 | 2 | 19 | 6 | 14 | 8 | 5 ]   (up to M0=32 neighbors)
+-- neighbors[1] -> [ 3 | 11 | 14 ]                          (up to M=16 neighbors)
+-- neighbors[2] -> [ 3 ]                                     (up to M=16 neighbors)
```

Why per-layer lists?
- Layer 0 allows `M0 = 2*M` connections for dense local search.
- Upper layers only allow `M` connections to stay sparse.
- Separate arrays keep memory tight and avoid wasting space.

---

### 5. `HNSWIndex`

**File:** `src/hnsw.h`, `src/hnsw.c`

The top-level container — the "database" that owns everything.

```c
typedef struct {
    /* Graph */
    HNSWNode **nodes;          // nodes[i] -> pointer to HNSWNode with id=i
    int        node_count;     // How many vectors are indexed
    int        node_capacity;  // Allocated size of nodes[]

    /* Vectors (parallel array to nodes[]) */
    Vector    *vectors;        // vectors[i] stores float data for node i

    /* Entry point */
    int        entry_point_id; // ID of the single node at the highest layer
    int        max_level;      // Current highest layer number in the graph

    /* Hyperparameters */
    int        dim;            // Vector dimensionality
    int        M;              // Max neighbors per layer (upper layers)
    int        M0;             // Max neighbors at layer 0 (= 2 * M)
    int        ef_construction;// Beam width during index build
    int        ef_search;      // Beam width during querying
    float      ml;             // Level multiplier = 1.0f / log(M)

    /* Pluggable distance metric */
    float    (*dist_fn)(const float*, const float*, int);
} HNSWIndex;
```

**Structural overview:**

```
HNSWIndex
+-- vectors[]   -> [ Vector | Vector | Vector | ... ]   (parallel array)
+-- nodes[]     -> [ *Node  | *Node  | *Node  | ... ]   (pointer array)
|                      |
|                   HNSWNode
|                   +-- neighbors[0] -> [adj list at layer 0]
|                   +-- neighbors[1] -> [adj list at layer 1]
|                   +-- ...
+-- entry_point_id = 42   (the single node at max_level)
+-- max_level  = 3
```

---

### 6. `KNNResult`

**File:** `src/hnsw.h`

The output of a query. A simple struct returned as a sorted array.

```c
typedef struct {
    int   id;        // Node ID of the neighbour
    float distance;  // Distance from query to this neighbour
} KNNResult;
// hnsw_search() returns KNNResult[k], sorted closest-first.
// Caller must free() it.
```

---

## High-Level Methodology

### Layer Structure

When a vector is inserted, it's randomly assigned a **maximum level** `L` using a geometric distribution:

```c
int level = (int)(-log((double)rand() / RAND_MAX) * ml);
```

This means:
- Almost all nodes land at layer 0 only.
- Far fewer reach layer 1.
- Even fewer reach layer 2.
- Extremely rarely, a node reaches layer 3+.

```
Layer   Approx. fraction of nodes
  0         100%   (all nodes)
  1          ~6%   (1/M of layer 0)
  2          ~0.4% (1/M^2 of layer 0)
  3          ~0.02%
```

The node at the current highest level becomes the **global entry point** for all future searches.

---

### Inserting a Vector

```
hnsw_insert(idx, new_vec, id)
     |
     +-- 1. Assign random level L = random_level(ml)
     |
     +-- 2. Phase A: Greedy descent from max_level -> L+1
     |      For each layer from max_level down to L+1:
     |        search_layer(new_vec, ep, ef=1, layer)
     |        ep <- single closest node found   <- entry point tightens each layer
     |
     +-- 3. Phase B: Insert & wire edges from L -> 0
     |      For each layer from min(L, max_level) down to 0:
     |        candidates <- search_layer(new_vec, ep, ef=ef_construction, layer)
     |        neighbors  <- select_neighbors(candidates, M or M0)
     |        add bidirectional edges: new_node <-> each neighbor
     |        for each neighbor: if degree > M -> prune its neighbor list
     |        ep <- closest node in candidates  <- entry point for next layer down
     |
     +-- 4. If L > max_level -> new node becomes entry_point_id
```

The two-phase design is intentional:
- **Phase A** quickly zooms into the right region at high layers (ef=1 = fast, coarse).
- **Phase B** carefully wires the new node into the graph with good neighbours (ef=ef_construction = thorough).

---

### Searching for Nearest Neighbours

```
hnsw_search(idx, query, k)
     |
     +-- 1. ep <- entry_point_id  (start at top of hierarchy)
     |
     +-- 2. Phase A: Greedy descent from max_level -> 1
     |      For each layer from max_level down to 1:
     |        search_layer(query, ep, ef=1, layer)
     |        ep <- closest node found
     |
     +-- 3. Phase B: Wide beam search at Layer 0
     |        W <- search_layer(query, ep, ef=max(ef_search, k), layer=0)
     |
     +-- 4. Extract top-K from W, sort ascending by distance
            Return KNNResult[k]
```

The key insight: all the precision work happens at layer 0, after the upper layers have "zoomed in" to roughly the right region of the vector space.

---

### The Core: `search_layer()`

This is called on every insert (Phase B) and every query (both phases). It implements **greedy beam search** within a single layer using the dual-heap structure:

```
search_layer(idx, query_vec, ep, ef, layer, visited):

  candidates  <- new Min-Heap    // nodes to explore, ordered by proximity
  results     <- new Max-Heap    // top-ef found so far (root = farthest)

  push ep into both heaps
  mark ep as visited

  while candidates is not empty:
    c <- candidates.pop_min()    // closest unvisited candidate
    f <- results.peek_max()      // farthest node in result set

    if dist(query, c) > dist(query, f):
      break   <- can't possibly improve results; stop

    for each neighbor e of c at this layer:
      if e is visited: skip
      mark e as visited
      d_e = dist(query, e)

      if d_e < dist(query, f)  OR  results.size < ef:
        push e into candidates  (min-heap)
        push e into results     (max-heap)
        if results.size > ef:
          results.pop_max()     <- prune farthest

  return results  (max-heap of ef nearest found)
```

**Why two heaps?**
- The min-heap tells you the *next best node to explore*.
- The max-heap tells you whether a new node is *worth adding to results*.
- The pruning condition (`dist(c) > dist(f)`) ensures you stop early rather than exploring the entire graph.

---

### Neighbour Selection

After `search_layer()` returns a candidate set, you select which nodes to actually connect as neighbours.

**Simple (required):**
Just pop the `M` closest items from the result heap.

**Heuristic (optional, improves quality):**
Prefer candidates that add *diversity* — a candidate is added only if it is closer to the query than to any already-selected neighbour. This prevents the graph from forming "cliques" and improves long-range navigability.

---

## How the Data Structures Connect

```
                   +-------------------+
                   |    HNSWIndex      |
                   |  vectors[]  ------+--> [ Vector | Vector | ... ]
                   |  nodes[]    ------+--> [ *HNSWNode | *HNSWNode | ... ]
                   |  entry_pt         |
                   +--------+----------+
                            | per node:
                   +--------v----------+
                   |    HNSWNode       |
                   |    level = 2      |
                   |    neighbors[] ---+--> [ layer0_list | layer1_list | layer2_list ]
                   +-------------------+

During search_layer():

  +--------------------+      +--------------------+
  | Min-Heap           |      | Max-Heap           |
  | (candidates)       |      | (results)          |
  | root = closest     |      | root = farthest    |
  | pop to explore     |      | pop to prune       |
  +--------------------+      +--------------------+

  +--------------------+
  | VisitedSet         |
  | gen[]              |
  | current_gen = N    |   O(1) reset per query
  +--------------------+
```

---

## Complexity Summary

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Insert** | O(M * log N) | One `search_layer` per layer, `log N` layers |
| **Query** | O(log N) | Upper layers greedy, O(1) amortised per layer |
| **Brute force** | O(N * dim) | Linear scan, no index |
| **Heap push/pop** | O(log ef) | ef = beam width (<= ef_construction or ef_search) |
| **VisitedSet reset** | O(1) | Generation counter increment |
| **select_neighbors** | O(ef * log ef) | Pop M items from a heap of ef items |

**Memory:** O(N * M * L) where L = average level height ~= log(N).

---

*Data Structures & Algorithms -- CS233AI, Semester III*
