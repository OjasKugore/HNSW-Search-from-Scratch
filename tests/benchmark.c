#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "workload.h"
#include "../src/vector.h"

enum { dimensions = 128, query_count = 100, neighbors = 10 };

static double monotonic_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}

static int benchmark_size(int count) {
    uint32_t data_seed = 2026, query_seed = 42;
    float vector[dimensions];
    float queries[query_count][dimensions];
    KNNResult expected[query_count][neighbors];
    const int widths[] = {50, 500, 1000};
    double exact_seconds = 0.0;

    utils_seed(12345);
    HNSWIndex *index = hnsw_create(dimensions, 16, 200, 50, dist_l2_sq);
    if (!index) return 0;

    double start = monotonic_seconds();
    for (int id = 0; id < count; id++) {
        fill_sample(vector, dimensions, &data_seed);
        if (!hnsw_insert(index, vector, id)) {
            hnsw_free(index);
            return 0;
        }
    }
    double build_seconds = monotonic_seconds() - start;
    for (int i = 0; i < query_count; i++) {
        fill_sample(queries[i], dimensions, &query_seed);
        start = monotonic_seconds();
        KNNResult *exact = brute_force_search(index, queries[i], neighbors);
        exact_seconds += monotonic_seconds() - start;
        if (!exact) {
            hnsw_free(index);
            return 0;
        }
        memcpy(expected[i], exact, sizeof(expected[i]));
        free(exact);
    }
    for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
        int matches = 0;
        double approx_seconds = 0.0;
        index->ef_search = widths[w];
        for (int i = 0; i < query_count; i++) {
            start = monotonic_seconds();
            KNNResult *found = hnsw_search(index, queries[i], neighbors);
            approx_seconds += monotonic_seconds() - start;
            if (!found) {
                hnsw_free(index);
                return 0;
            }
            matches += count_matches(found, expected[i], neighbors);
            free(found);
        }
        printf("%d\t%d\t%.3f\t%.2f\t%.2f\t%.2f\t%.3f\n",
               count, widths[w], build_seconds, exact_seconds * 1e6 / query_count,
               approx_seconds * 1e6 / query_count, exact_seconds / approx_seconds,
               (double)matches / (neighbors * query_count));
        fflush(stdout);
    }
    hnsw_free(index);
    return 1;
}

int main(int argc, char **argv) {
    int sizes[] = {1000, 10000, 100000};
    int runs = 3;
    if (argc == 2) {
        char *end;
        errno = 0;
        long count = strtol(argv[1], &end, 10);
        if (errno || *end || end == argv[1] || count < neighbors || count > INT_MAX) {
            fprintf(stderr, "Specify an integer size between %d and %d.\n", neighbors, INT_MAX);
            return EXIT_FAILURE;
        }
        sizes[0] = (int)count;
        runs = 1;
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [dataset-size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    puts("Squared L2, dim=128, M=16, ef_construction=200, k=10, queries=100");
    puts("Seeds: vectors=2026, queries=42, levels=12345");
    puts("N\tef_search\tbuild_s\texact_us/query\thnsw_us/query\tspeedup\trecall@10");
    for (int i = 0; i < runs; i++) {
        if (!benchmark_size(sizes[i])) {
            fprintf(stderr, "Benchmark failed for N=%d.\n", sizes[i]);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
