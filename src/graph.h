#ifndef GRAPH_H
#define GRAPH_H

enum { HNSW_MAX_LEVEL = 16, HNSW_MAX_M = 256 };

typedef struct {
    int id;
    int level;
    int **neighbors;
    int *n_count;
    int *n_cap;
    char *label;
} HNSWNode;

HNSWNode *node_create(int id, int level, int M, int M0);
void node_free(HNSWNode *node);
/* Each layer reserves one extra slot for insertion before pruning. */
int node_add_neighbor(HNSWNode *node, int layer, int id);
void node_remove_neighbor(HNSWNode *node, int layer, int id);
void node_print(const HNSWNode *node);

#endif
