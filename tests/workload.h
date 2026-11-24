#ifndef TESTS_WORKLOAD_H
#define TESTS_WORKLOAD_H

#include <stdint.h>

#include "../src/hnsw.h"

/* Keep fixture coordinates independent of random level selection in the index. */
static void fill_sample(float *values, int dim, uint32_t *state) {
    for (int i = 0; i < dim; i++) {
        *state = UINT32_C(1664525) * *state + UINT32_C(1013904223);
        values[i] = (float)(*state >> 8) / 16777216.0f;
    }
}

static int count_matches(const KNNResult *found, const KNNResult *exact, int k) {
    int matches = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (found[i].id == exact[j].id) {
                matches++;
                break;
            }
        }
    }
    return matches;
}

#endif
