# HNSW — Data Structures & Methodology

> Design notes for the course project. The [project guide](../README.md) and [public header](../src/hnsw.h) describe the implemented interface.

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

The graph can reduce the number of distance calculations. Exact search evaluates all vectors.

---

## Data Structures

### 1. `Vector`

**File:** `src/vector.h`, `src/vector.c`

A standalone vector stores its ID, dimension, and float coordinates in one allocation.

```c
typedef struct {
    int id;
    int dim;
    float data[];
} Vector;
```

The index copies coordinates into a flat array. Each graph node can own a word label.

**Distance functions** live here too:
```c
float dist_l2_sq(const float *a, const float *b, int dim);         // ||a - b||^2
float dist_cosine(const float *a, const float *b, int dim);        // 1 - cos(a,b)
float dist_dot(const float *a, const float *b, int dim);           // dot product
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
    int       is_min;   // HEAP_MIN (1) or HEAP_MAX (0)
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
    unsigned int *gen; // gen[node_id] = generation when this node was last visited
    unsigned int current; // The current query generation number
    int  capacity;    // Total number of nodes tracked
} VisitedSet;
```

**Why not just use `bool visited[N]`?**

Clearing a `bool` array is **O(N)** per query — catastrophic for 100K+ nodes. The generation trick avoids this:

```
Query 1: current = 1
  -> Mark node 5: gen[5] = 1
  -> Mark node 9: gen[9] = 1

Query 2: visited_reset() -> current = 2   <- O(1)!
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
    int  *n_count;      // Number of valid entries in neighbors[layer]
    int  *n_cap;   // Allocated capacity of neighbors[layer]
    char *label;               // Optional word label owned by the node
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
    float     *raw_vectors;    // Vector i starts at raw_vectors + (size_t)i * dim

    /* Entry point */
    int        entry_point; // ID of the chosen node at the highest layer
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
    VisitedSet *visited;       // Scratch space shared by searches on this index
} HNSWIndex;
```

**Structural overview:**

```
HNSWIndex
+-- raw_vectors -> [ row 0 coordinates | row 1 coordinates | ... ]
+-- nodes[]     -> [ *Node  | *Node  | *Node  | ... ]   (pointer array)
|                      |
|                   HNSWNode
|                   +-- neighbors[0] -> [adj list at layer 0]
|                   +-- neighbors[1] -> [adj list at layer 1]
|                   +-- ...
+-- entry_point = 42   (the entry node at max_level)
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
     +-- 4. If L > max_level -> new node becomes entry_point
```

The two-phase design is intentional:
- **Phase A** quickly zooms into the right region at high layers (ef=1 = fast, coarse).
- **Phase B** carefully wires the new node into the graph with good neighbours (ef=ef_construction = thorough).

---

### Searching for Nearest Neighbours

```
hnsw_search(idx, query, k)
     |
     +-- 1. ep <- entry_point  (start at top of hierarchy)
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

    if results.size == ef and dist(query, c) > dist(query, f):
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

Insertion considers the candidates in order of distance. It prefers a candidate when no selected neighbor is closer to that candidate than the query is. The closest remaining candidates fill spare slots.

The new node connects to each selected neighbor in both directions. When a list exceeds its limit, the index removes the farthest eligible link from both endpoints. Layer zero always links consecutive node IDs. These protected links keep every vector reachable, including duplicate vectors.

---

## How the Data Structures Connect

```
                   +-------------------+
                   |    HNSWIndex      |
                   |  raw_vectors -----+--> [ row 0 floats | row 1 floats | ... ]
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
  | current = N    |   O(1) reset per query
  +--------------------+
```

---

## Complexity Summary

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Insert** | Data-dependent | Layer count, construction width, and edge selection affect cost |
| **Query** | Data-dependent | Graph traversal can visit every vector |
| **Brute force** | O(N × (dim + log(k + 1))) | Scan all vectors and retain the best k in a heap |
| **Heap push/pop** | O(log H) | H is the heap size. The retained result heap contains at most ef items |
| **VisitedSet reset** | Usually O(1) | Counter wraparound requires O(N) to clear the marks |
| **Neighbor selection** | O(ef × (log ef + M × dim)) | Sort candidates and check distances to selected neighbors |

**Memory:** Vectors use O(N × dim) storage. Graph storage is O(N × M × L), where L is the average number of layers per node.

---

*Data Structures & Algorithms -- CS233AI, Semester III*
