#include "graph.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

HNSWNode *node_create(int id, int level, int M, int M0) {
    if (id < 0 || level < 0 || level > HNSW_MAX_LEVEL || M < 1 || M0 < 1 ||
        M == INT_MAX || M0 == INT_MAX) return NULL;

    size_t layers = (size_t)level + 1;
    HNSWNode *node = calloc(1, sizeof(*node));
    if (!node) return NULL;
    node->id = id;
    node->level = level;
    node->neighbors = calloc(layers, sizeof(*node->neighbors));
    node->n_count = calloc(layers, sizeof(*node->n_count));
    node->n_cap = malloc(layers * sizeof(*node->n_cap));
    if (!node->neighbors || !node->n_count || !node->n_cap) goto fail;

    for (int layer = 0; layer <= level; ++layer) {
        int slots = (layer == 0 ? M0 : M) + 1;
        if ((size_t)slots > SIZE_MAX / sizeof(int)) goto fail;
        node->neighbors[layer] = malloc((size_t)slots * sizeof(int));
        if (!node->neighbors[layer]) goto fail;
        node->n_cap[layer] = slots;
    }
    return node;
fail:
    node_free(node);
    return NULL;
}

void node_free(HNSWNode *node) {
    if (!node) return;
    if (node->neighbors) {
        for (int layer = 0; layer <= node->level; ++layer) free(node->neighbors[layer]);
    }
    free(node->neighbors);
    free(node->n_count);
    free(node->n_cap);
    free(node->label);
    free(node);
}

int node_add_neighbor(HNSWNode *node, int layer, int id) {
    if (!node || layer < 0 || layer > node->level || id < 0 || id == node->id)
        return 0;
    int *ids = node->neighbors[layer];
    int count = node->n_count[layer];
    for (int i = 0; i < count; ++i) {
        if (ids[i] == id) return 1;
    }
    if (count >= node->n_cap[layer]) return 0;
    ids[node->n_count[layer]++] = id;
    return 1;
}

void node_remove_neighbor(HNSWNode *node, int layer, int id) {
    if (!node || layer < 0 || layer > node->level) return;
    int *ids = node->neighbors[layer];
    for (int i = 0; i < node->n_count[layer]; ++i) {
        if (ids[i] == id) {
            ids[i] = ids[--node->n_count[layer]];
            return;
        }
    }
}

void node_print(const HNSWNode *node) {
    if (!node) return;
    for (int layer = node->level; layer >= 0; --layer) {
        printf("Node %d, layer %d:", node->id, layer);
        for (int i = 0; i < node->n_count[layer]; ++i)
            printf(" %d", node->neighbors[layer][i]);
        putchar('\n');
    }
}
