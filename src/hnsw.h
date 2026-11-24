#ifndef HNSW_H
#define HNSW_H

#include "graph.h"
#include "heap.h"
#include "utils.h"
#include "vector.h"

typedef float (*DistanceFn)(const float *, const float *, int);

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

typedef struct {
    int id;
    float distance;
} KNNResult;

/* Smaller metric values rank first. Require finite coordinates and 2 <= M <= 256. */
HNSWIndex *hnsw_create(int dim, int M, int ef_construction, int ef_search,
                       DistanceFn dist_fn);
int hnsw_reserve(HNSWIndex *idx, int capacity);
void hnsw_free(HNSWIndex *idx);
void hnsw_print_stats(const HNSWIndex *idx);

/* IDs are consecutive from zero. A successful insertion copies the vector. */
int hnsw_insert(HNSWIndex *idx, const float *vec, int id);
/* Both functions return k caller-owned results, or NULL on failure.
   Require 1 <= k <= node_count. Searches on one index share visited marks. */
KNNResult *hnsw_search(HNSWIndex *idx, const float *query, int k);
KNNResult *brute_force_search(const HNSWIndex *idx, const float *query, int k);
int hnsw_save(const HNSWIndex *idx, const char *path);
HNSWIndex *hnsw_load(const char *path);

#endif
