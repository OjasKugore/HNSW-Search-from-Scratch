#include "hnsw.h"
#include "heap.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float *vector_at(const HNSWIndex *idx, int id) {
    return idx->raw_vectors + (size_t)id * (size_t)idx->dim;
}

static int finite_vector(const float *vec, int dim) {
    if (!vec) return 0;
    for (int i = 0; i < dim; ++i) {
        if (!isfinite(vec[i])) return 0;
    }
    return 1;
}

static HeapItem measure(const HNSWIndex *idx, const float *query, int id) {
    float d = idx->dist_fn(query, vector_at(idx, id), idx->dim);
    return (HeapItem){id, isnan(d) ? INFINITY : d};
}

static int nearer(HeapItem a, HeapItem b) {
    return a.distance < b.distance || (a.distance == b.distance && a.node_id < b.node_id);
}

HNSWIndex *hnsw_create(int dim, int M, int ef_construction, int ef_search,
                       DistanceFn dist_fn) {
    if (dim < 1 || M < 2 || M > HNSW_MAX_M || ef_construction < 1 ||
        ef_search < 1 || !dist_fn || (size_t)dim > SIZE_MAX / sizeof(float))
        return NULL;

    HNSWIndex *idx = calloc(1, sizeof(*idx));
    if (!idx) return NULL;
    idx->visited = visited_create(0);
    idx->frontier = heap_create(ef_construction, HEAP_MIN);
    idx->results = heap_create(ef_construction, HEAP_MAX);
    if (!idx->visited || !idx->frontier || !idx->results) {
        hnsw_free(idx);
        return NULL;
    }
    idx->dim = dim;
    idx->M = M;
    idx->M0 = M * 2;
    idx->ef_construction = ef_construction;
    idx->ef_search = ef_search;
    idx->ml = 1.0f / logf((float)M);
    idx->entry_point = -1;
    idx->max_level = -1;
    idx->dist_fn = dist_fn;
    return idx;
}

int hnsw_reserve(HNSWIndex *idx, int capacity) {
    if (!idx || capacity < 0) return 0;
    if (capacity <= idx->node_capacity) return 1;
    if ((size_t)capacity > SIZE_MAX / sizeof(*idx->nodes) ||
        (size_t)capacity > SIZE_MAX / sizeof(float) / (size_t)idx->dim)
        return 0;

    if (!visited_reserve(idx->visited, capacity)) return 0;
    HNSWNode **nodes = realloc(idx->nodes, (size_t)capacity * sizeof(*nodes));
    if (!nodes) return 0;
    idx->nodes = nodes;
    size_t elements = (size_t)capacity * (size_t)idx->dim;
    float *vectors = realloc(idx->raw_vectors, elements * sizeof(*vectors));
    if (!vectors) return 0;
    idx->raw_vectors = vectors;
    idx->node_capacity = capacity;
    return 1;
}

void hnsw_free(HNSWIndex *idx) {
    if (!idx) return;
    for (int id = 0; id < idx->node_count; ++id) node_free(idx->nodes[id]);
    free(idx->nodes);
    free(idx->raw_vectors);
    visited_free(idx->visited);
    heap_free(idx->frontier);
    heap_free(idx->results);
    free(idx);
}

void hnsw_print_stats(const HNSWIndex *idx) {
    if (!idx) return;
    printf("Nodes: %d, dimensions: %d, M: %d, M0: %d, entry: %d, top layer: %d\n",
           idx->node_count, idx->dim, idx->M, idx->M0, idx->entry_point, idx->max_level);
    for (int layer = idx->max_level; layer >= 0; --layer) {
        int count = 0;
        long long links = 0;
        for (int id = 0; id < idx->node_count; ++id) {
            HNSWNode *node = idx->nodes[id];
            if (node->level < layer) continue;
            ++count;
            links += node->n_count[layer];
        }
        printf("Layer %d: %d nodes, %.2f neighbors per node\n", layer, count,
               count ? (double)links / count : 0.0);
    }
}

/* Greedy descent is a width-one search. It needs no heaps or visited array. */
static int descend(const HNSWIndex *idx, const float *query, int stop_layer) {
    HeapItem best = measure(idx, query, idx->entry_point);
    for (int layer = idx->max_level; layer > stop_layer; --layer) {
        int previous;
        do {
            previous = best.node_id;
            HNSWNode *node = idx->nodes[previous];
            for (int i = 0; i < node->n_count[layer]; ++i) {
                HeapItem next = measure(idx, query, node->neighbors[layer][i]);
                if (nearer(next, best)) best = next;
            }
        } while (previous != best.node_id);
    }
    return best.node_id;
}

/* The frontier expands the closest candidate. The bounded results heap exposes
   the farthest retained item, which sets the cutoff for new candidates. */
static Heap *search_layer(HNSWIndex *idx, const float *query, int entry, int ef, int layer) {
    if (ef > idx->node_count) ef = idx->node_count;
    Heap *frontier = idx->frontier;
    Heap *results = idx->results;
    if (!frontier || !results) return NULL;
    if (!heap_ensure_capacity(frontier, ef) || !heap_ensure_capacity(results, ef)) return NULL;
    heap_clear(frontier);
    heap_clear(results);

    VisitedSet *vs = idx->visited;
    visited_reset(vs);
    vs->gen[entry] = vs->current;
    HeapItem start = measure(idx, query, entry);
    heap_push(frontier, entry, start.distance);
    heap_push(results, entry, start.distance);
    while (!heap_is_empty(frontier)) {
        HeapItem candidate = heap_pop(frontier);
        if (results->size == ef && candidate.distance > results->items[0].distance) break;
        HNSWNode *node = idx->nodes[candidate.node_id];
        for (int i = 0; i < node->n_count[layer]; ++i) {
            int id = node->neighbors[layer][i];
            if (vs->gen[id] == vs->current) continue;
            vs->gen[id] = vs->current;
            HeapItem item = measure(idx, query, id);
            if (results->size == ef) {
                if (!nearer(item, results->items[0])) continue;
                heap_replace_root(results, id, item.distance);
            } else {
                heap_push(results, id, item.distance);
            }
            if (!heap_try_push(frontier, id, item.distance)) {
                return NULL;
            }
        }
    }
    return results;
}

/* Prefer candidates closer to the query than to any selected neighbor.
   Keep the base-layer predecessor even when many vectors are identical. */
static int choose_diverse(const HNSWIndex *idx, HeapItem *sorted, int count,
                          int *chosen, int predecessor, int limit) {
    int size = 0;
    if (predecessor >= 0) chosen[size++] = predecessor;
    int deferred = 0;
    for (int i = 0; i < count && size < limit; ++i) {
        HeapItem candidate = sorted[i];
        if (candidate.node_id == predecessor) continue;
        const float *vec = vector_at(idx, candidate.node_id);
        int diverse = 1;
        for (int j = 0; j < size; ++j) {
            if (measure(idx, vec, chosen[j]).distance < candidate.distance) {
                diverse = 0;
                break;
            }
        }
        if (diverse) chosen[size++] = candidate.node_id;
        else sorted[deferred++] = candidate;
    }
    for (int i = 0; i < deferred && size < limit; ++i) chosen[size++] = sorted[i].node_id;
    return size;
}

/* Fill the new node's lists before changing any existing edge. */
static int select_neighbors(HNSWIndex *idx, const float *vec, HNSWNode *node) {
    int entry = descend(idx, vec, node->level);
    int top = node->level < idx->max_level ? node->level : idx->max_level;
    for (int layer = top; layer >= 0; --layer) {
        Heap *results = search_layer(idx, vec, entry, idx->ef_construction, layer);
        if (!results) return 0;
        int count;
        /* A max-heap drains into its vacated tail from right to left. */
        heap_drain_sorted(results, results->items, &count);
        entry = results->items[0].node_id;
        int *ids = node->neighbors[layer];
        int predecessor = layer == 0 ? node->id - 1 : -1;
        int limit = layer == 0 ? idx->M0 : idx->M;
        node->n_count[layer] = choose_diverse(idx, results->items, count, ids, predecessor, limit);
    }
    return 1;
}

static void prune_one(HNSWIndex *idx, HNSWNode *node, int layer) {
    int limit = layer == 0 ? idx->M0 : idx->M;
    if (node->n_count[layer] <= limit) return;
    const float *origin = vector_at(idx, node->id);
    HeapItem farthest = {-1, -INFINITY};
    for (int i = 0; i < node->n_count[layer]; ++i) {
        int id = node->neighbors[layer][i];
        if (layer == 0 && abs(id - node->id) == 1) continue;
        HeapItem item = measure(idx, origin, id);
        if (farthest.node_id < 0 || nearer(farthest, item)) farthest = item;
    }
    node_remove_neighbor(node, layer, farthest.node_id);
    node_remove_neighbor(idx->nodes[farthest.node_id], layer, node->id);
}

int hnsw_insert(HNSWIndex *idx, const float *vec, int id) {
    if (!idx || id != idx->node_count || id == INT_MAX || !finite_vector(vec, idx->dim))
        return 0;
    int level = random_level(idx->ml, HNSW_MAX_LEVEL);
    HNSWNode *node = node_create(id, level, idx->M, idx->M0);
    if (!node) return 0;
    if (id && !select_neighbors(idx, vec, node)) goto fail;

    /* A caller may insert a vector already stored in the index. Keep it valid
       across the realloc that grows the flat vector array. */
    size_t bytes = (size_t)idx->dim * sizeof(float);
    if (id == idx->node_capacity) {
        int capacity = id > INT_MAX / 2 ? INT_MAX : (id ? 2 * id : 16);
        float *copy = malloc(bytes);
        if (!copy) goto fail;
        memcpy(copy, vec, bytes);
        if (!hnsw_reserve(idx, capacity)) {
            free(copy);
            goto fail;
        }
        memcpy(idx->raw_vectors + (size_t)id * (size_t)idx->dim, copy, bytes);
        free(copy);
    } else {
        memmove(idx->raw_vectors + (size_t)id * (size_t)idx->dim, vec, bytes);
    }
    idx->nodes[id] = node;
    ++idx->node_count;
    /* Iterate backwards: pruning a reverse link can remove the pending neighbor
       from this list and move the last (already processed) ID into its place. */
    for (int layer = 0; layer <= level; ++layer) {
        for (int i = node->n_count[layer] - 1; i >= 0; --i) {
            HNSWNode *neighbor = idx->nodes[node->neighbors[layer][i]];
            node_add_neighbor(neighbor, layer, id);
            prune_one(idx, neighbor, layer);
        }
    }
    if (level > idx->max_level) {
        idx->max_level = level;
        idx->entry_point = id;
    }
    return 1;
fail:
    node_free(node);
    return 0;
}

static KNNResult *take_best(Heap *results, int k) {
    if (!results) return NULL;
    KNNResult *out = NULL;
    if (results->size >= k && (size_t)k <= SIZE_MAX / sizeof(*out))
        out = malloc((size_t)k * sizeof(*out));
    if (out) {
        while (results->size > k) heap_pop(results);
        for (int i = k - 1; i >= 0; --i) {
            HeapItem item = heap_pop(results);
            out[i] = (KNNResult){item.node_id, item.distance};
        }
    }
    return out;
}

KNNResult *hnsw_search(HNSWIndex *idx, const float *query, int k) {
    if (!idx || k < 1 || k > idx->node_count || !finite_vector(query, idx->dim))
        return NULL;
    int entry = descend(idx, query, 0);
    int ef = idx->ef_search > k ? idx->ef_search : k;
    return take_best(search_layer(idx, query, entry, ef, 0), k);
}

KNNResult *brute_force_search(const HNSWIndex *idx, const float *query, int k) {
    if (!idx || k < 1 || k > idx->node_count || !finite_vector(query, idx->dim))
        return NULL;
    Heap *results = heap_create(k, HEAP_MAX);
    if (!results) return NULL;
    for (int id = 0; id < idx->node_count; ++id) {
        HeapItem item = measure(idx, query, id);
        if (results->size == k) {
            if (!nearer(item, results->items[0])) continue;
            heap_replace_root(results, id, item.distance);
        } else {
            heap_push(results, id, item.distance);
        }
    }
    KNNResult *out = take_best(results, k);
    heap_free(results);
    return out;
}
