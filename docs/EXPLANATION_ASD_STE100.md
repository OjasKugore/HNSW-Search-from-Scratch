# HNSW Vector Search Engine: Methodology, Data Structures, and Dry Run

This document explains the Hierarchical Navigable Small World (HNSW) search engine in this repository. It follows the ASD-STE100 Simplified Technical English standard: short sentences, active voice, consistent terms, and one instruction per step.

---

## 1. Overview

The HNSW engine finds the `k` closest vectors to a query vector in a large dataset. A full scan compares the query vector against every stored vector. That scan is slow on large datasets. The HNSW index arranges all vectors into a multi-layer graph so a search visits only a small subset of nodes.

- **Layer 0** contains every vector in the index and has the most links per node.
- **Upper layers** (`Layer 1` to `Layer 16`) contain fewer nodes and have longer links across the dataset.
- A search starts at the top layer to move near the target quickly.
- The search then moves down to Layer 0 to find the closest vectors.

---

## 2. Data Structures

The C implementation uses six core data structures. Each structure has one specific task.

### 2.1 `Vector` ([src/vector.h](../src/vector.h), [src/vector.c](../src/vector.c))

The `Vector` structure holds one vector and its dimension count in one block of memory.

```c
typedef struct {
    int id;
    int dim;
    float data[];
} Vector;
```

- `id`: Identifies the vector with an integer number.
- `dim`: Stores the number of coordinates in the vector.
- `data[]`: Stores the coordinate values directly after the structure header.

Three functions calculate distances between two coordinate arrays:

- `dist_l2_sq`: Calculates the squared Euclidean distance.
- `dist_cosine`: Calculates the cosine distance (`1 - cosine_similarity`).
- `dist_dot`: Calculates the dot product.

### 2.2 `HNSWNode` ([src/graph.h](../src/graph.h), [src/graph.c](../src/graph.c))

Each `HNSWNode` structure represents one vertex in the multi-layer graph.

```c
typedef struct {
    int id;
    int level;
    int **neighbors;
    int *n_count;
    int *n_cap;
    char *label;
} HNSWNode;
```

- `id`: Stores the sequential node ID (`0` to `node_count - 1`).
- `level`: Stores the highest layer that contains this node (`0` to `HNSW_MAX_LEVEL`, which is `16`).
- `neighbors`: Points to one integer array of neighbor IDs for each layer (`0` to `level`).
- `n_count`: Stores the current number of links at each layer.
- `n_cap`: Stores the maximum number of link slots at each layer (`M0 + 1` at Layer 0, `M + 1` at upper layers).
- `label`: Stores an optional word string (or `NULL` for numeric vectors).

### 2.3 `Heap` and `HeapItem` ([src/heap.h](../src/heap.h), [src/heap.c](../src/heap.c))

The `Heap` structure implements a dynamic binary heap of `HeapItem` records.

```c
typedef struct {
    int node_id;
    float distance;
} HeapItem;

typedef struct {
    int size;
    int capacity;
    int is_min;
    HeapItem *items;
} Heap;
```

- `is_min = 1` (`HEAP_MIN`): Keeps the item with the smallest distance at `items[0]`. The search uses this min-heap as the `frontier` queue to pick the closest candidate next.
- `is_min = 0` (`HEAP_MAX`): Keeps the item with the largest distance at `items[0]`. The search uses this max-heap as the `results` buffer of size `ef` so `items[0]` gives the distance cutoff.
- `heap_replace_root`: Replaces `items[0]` and moves the new item down in one pass.
- `heap_clear`: Sets `size` to `0` so the index can reuse the allocated array.

### 2.4 `VisitedSet` ([src/utils.h](../src/utils.h), [src/utils.c](../src/utils.c))

The `VisitedSet` structure tracks which nodes a layer search has already checked.

```c
typedef struct {
    unsigned int *gen;
    unsigned int current;
    int capacity;
} VisitedSet;
```

- `gen`: Stores one unsigned integer stamp for each node ID.
- `current`: Stores the active search stamp.
- `visited_reset`: Increases `current` by `1` to clear all marks in constant time ($O(1)$). It clears the `gen` array with `memset` only when `current` wraps around to `0`.

### 2.5 `HNSWIndex` ([src/hnsw.h](../src/hnsw.h), [src/hnsw.c](../src/hnsw.c))

The `HNSWIndex` structure owns the full graph, all raw vectors, and the shared scratch buffers.

```c
typedef struct {
    HNSWNode **nodes;
    int node_count;
    int node_capacity;
    float *raw_vectors;
    int entry_point;
    int max_level;
    int dim;
    int M;
    int M0;
    int ef_construction;
    int ef_search;
    float ml;
    DistanceFn dist_fn;
    VisitedSet *visited;
    Heap *frontier;
    Heap *results;
} HNSWIndex;
```

- `nodes`: Points to the array of `HNSWNode` pointers.
- `raw_vectors`: Stores all vector coordinates in one continuous float array for fast memory access. The coordinates for node `id` start at offset `id * dim`.
- `entry_point`: Stores the ID of the node at the top of the graph.
- `max_level`: Stores the highest active layer in the index.
- `M` and `M0`: Set the link limits per node (`M0 = 2 * M` for Layer 0).
- `ef_construction`: Sets the candidate list size during insertion.
- `ef_search`: Sets the candidate list size during search.
- `ml`: Stores `1.0f / logf((float)M)` for random layer selection.
- `visited`, `frontier`, `results`: Provide reusable memory buffers so searches do not allocate heaps repeatedly.

### 2.6 `KNNResult` ([src/hnsw.h](../src/hnsw.h))

```c
typedef struct {
    int id;
    float distance;
} KNNResult;
```

- `hnsw_search` returns a caller-owned array of `k` `KNNResult` items sorted from smallest distance to largest distance.

---

## 3. Methodology

### 3.1 Insert a Vector (`hnsw_insert`)

1. Validate that `id` equals `node_count` and that all vector coordinates are finite numbers.
2. Call `random_level` to assign an integer `level` from an exponential decay distribution.
3. Allocate a new `HNSWNode` for `id` and `level`.
4. If the index is not empty, move down from `max_level` to `node->level + 1` using `descend`. At each upper layer, jump greedily to the closest neighbor until no neighbor is closer.
5. For each layer from `min(node->level, max_level)` down to `0`, run `search_layer` with width `ef_construction` to collect candidate neighbors.
6. Sort the candidates in `results` with `heap_drain_sorted`.
7. Run `choose_diverse` to select up to `M` links (or `M0` links at Layer 0). This function keeps candidates that are closer to the new vector than to any already-selected neighbor. At Layer 0, it also keeps predecessor `id - 1` connected.
8. Copy the new coordinates into `raw_vectors` and store the node pointer in `nodes[id]`.
9. Add reverse links from each selected neighbor back to `id`.
10. If a neighbor has too many links, run `prune_one`. This removes its farthest link (except `id - 1` at Layer 0).
11. If `level` is greater than `max_level`, set `max_level` to `level` and `entry_point` to `id`.

### 3.2 Search for Nearest Neighbors (`hnsw_search`)

1. Validate the index, `k`, and the query coordinates.
2. Call `descend` to walk greedily from `entry_point` at `max_level` down to Layer 1. This step uses no heaps and returns the closest node found as the Layer 0 entry point.
3. Set the beam width `ef` to the larger of `ef_search` and `k`.
4. Call `search_layer` at Layer 0:
   - Reset `VisitedSet` in $O(1)$ time and mark the entry node as visited.
   - Push the entry node and its distance into `frontier` (min-heap) and `results` (max-heap).
   - Pop the closest node `candidate` from `frontier`.
   - Stop the loop if `results` has `ef` items and `candidate.distance` is greater than `results->items[0].distance`.
   - Check each unvisited neighbor of `candidate` at Layer 0. Mark the neighbor as visited and calculate its distance to the query.
   - If `results` has fewer than `ef` items, push the neighbor into both `frontier` and `results`.
   - If `results` is full and the neighbor is closer than `results->items[0]`, replace the root of `results`. Then push the neighbor into `frontier`.
5. Call `take_best` to pop extra items from `results` until `k` items remain.
6. Pop those `k` items in reverse order into the output `KNNResult` array.

---

## 4. Dry Run Example

### 4.1 Sample Index Setup

This example uses **6 nodes** in 2D space with squared Euclidean distance (`dist_l2_sq`). Set `M = 2`, `M0 = 4`, `ef_search = 3`, and `k = 2`.

| Node ID | Coordinates `(x, y)` | Max Layer | Distance to `Q = (6.5, 4.5)` |
|---:|---|---:|---:|
| `0` | `(0, 0)` | `2` (`entry_point`) | `(6.5 - 0)^2 + (4.5 - 0)^2 = 62.50` |
| `1` | `(8, 0)` | `0` | `(6.5 - 8)^2 + (4.5 - 0)^2 = 22.50` |
| `2` | `(4, 3)` | `1` | `(6.5 - 4)^2 + (4.5 - 3)^2 = 8.50` |
| `3` | `(10, 8)` | `2` | `(6.5 - 10)^2 + (4.5 - 8)^2 = 24.50` |
| `4` | `(7, 6)` | `0` | `(6.5 - 7)^2 + (4.5 - 6)^2 = 2.50` |
| `5` | `(6, 5)` | `1` | `(6.5 - 6)^2 + (4.5 - 5)^2 = 0.50` |

**Graph links per layer:**

- **Layer 2:** `0 <-> 3`
- **Layer 1:** `0 <-> 2`, `2 <-> 5`, `5 <-> 3`, `2 <-> 3`
- **Layer 0:** `5` links to `[2, 4, 3, 1]`; `4` links to `[1, 3, 5, 2]`; `2` links to `[0, 1, 4, 5]`

### 4.2 Step-by-Step Search Execution for `Q = (6.5, 4.5)`

#### Phase 1: Greedy Descent (`descend`)

1. Start at `entry_point = 0` on **Layer 2**. `best` is `(Node 0, 62.50)`.
2. Check Layer 2 neighbor `3` (`distance = 24.50`). Because `24.50 < 62.50`, set `best` to `(Node 3, 24.50)`.
3. Check Layer 2 neighbor of `3`, which is `0` (`62.50`). No neighbor is closer. Drop to **Layer 1** at `Node 3`.
4. At **Layer 1**, check neighbors of `3`: `5` (`distance = 0.50`) and `2` (`distance = 8.50`).
5. Set `best` to `(Node 5, 0.50)`.
6. Check Layer 1 neighbors of `5` (`2` at `8.50` and `3` at `24.50`). No neighbor is closer. Drop to **Layer 0** with `entry = 5`.

#### Phase 2: Layer 0 Beam Search (`search_layer` with `ef = 3`)

1. Call `visited_reset` to increment `vs->current` to the next generation. Mark `Node 5` as visited.
2. Push `(5, 0.50)` into `frontier` (min-heap) and `results` (max-heap).
   - `frontier`: `[(5, 0.50)]`
   - `results`: `[(5, 0.50)]`
3. **Iteration 1:** Pop `(5, 0.50)` from `frontier`. Check its Layer 0 neighbors `[2, 4, 3, 1]`:
   - `Node 2` (`8.50`): Mark visited. `results` has fewer than `3` items, so push `(2, 8.50)` to both heaps.
   - `Node 4` (`2.50`): Mark visited. `results` has fewer than `3` items, so push `(4, 2.50)` to both heaps. Now `results` is full with root `(2, 8.50)`.
   - `Node 3` (`24.50`): Mark visited. `24.50 > 8.50`, so ignore `Node 3`.
   - `Node 1` (`22.50`): Mark visited. `22.50 > 8.50`, so ignore `Node 1`.
   - Current state: `frontier` = `[(4, 2.50), (2, 8.50)]`, `results` root = `(2, 8.50)`.
4. **Iteration 2:** Pop `(4, 2.50)` from `frontier`. Check cutoff: `2.50 <= 8.50`, so continue.
   - Check neighbors `[1, 3, 5, 2]`. `VisitedSet` already marks all four nodes, so skip them without distance calculations.
5. **Iteration 3:** Pop `(2, 8.50)` from `frontier`. Check cutoff: `8.50 <= 8.50`, so continue.
   - Check neighbors `[0, 1, 4, 5]`. `Node 0` (`62.50`) is greater than `8.50`, so ignore it. Nodes `1`, `4`, and `5` are already visited.
6. `frontier` is now empty. Stop the search loop.

#### Phase 3: Extract Results (`take_best`)

1. `results` contains `[(2, 8.50), (4, 2.50), (5, 0.50)]`.
2. Pop `1` item from `results` (`(2, 8.50)`) until `size` equals `k = 2`.
3. Pop the remaining `2` items in reverse order into the output array:
   - `out[0]` = `{ id: 5, distance: 0.50 }`
   - `out[1]` = `{ id: 4, distance: 2.50 }`

The search returns the 2 closest vectors (`Node 5` and `Node 4`). `VisitedSet` and the max-heap cutoff prevent extra distance checks.
