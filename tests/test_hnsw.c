#define _POSIX_C_SOURCE 200809L
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "workload.h"
#include "../src/graph.h"
#include "../src/heap.h"
#include "../src/utils.h"
#include "../src/vector.h"

static void test_vector(void) {
    const float values[] = {1.25f, -2.5f, 3.75f};
    Vector *vector = vector_create(3);
    assert(vector != NULL);
    assert(vector->dim == 3);
    assert(vector->id == -1);

    for (int i = 0; i < vector->dim; i++) {
        vector_set(vector, i, values[i]);
    }
    for (int i = 0; i < vector->dim; i++) {
        assert(vector_get(vector, i) == values[i]);
    }

    vector_fill_random(vector);
    for (int i = 0; i < vector->dim; i++) {
        float value = vector_get(vector, i);
        assert(value >= 0.0f && value <= 1.0f);
    }
    vector_free(vector);
}

static void test_distances(void) {
    const float a[] = {3.0f, 4.0f};
    const float b[] = {-4.0f, 3.0f};
    const float opposite[] = {-6.0f, -8.0f};
    const float zero[] = {0.0f, 0.0f};

    assert(dist_l2_sq(a, a, 2) == 0.0f);
    assert(dist_l2_sq(a, b, 2) == 50.0f);
    assert(dist_dot(a, b, 2) == 0.0f);
    assert(dist_dot(a, opposite, 2) == -50.0f);
    assert(fabsf(dist_cosine(a, a, 2)) < 1e-6f);
    assert(fabsf(dist_cosine(a, b, 2) - 1.0f) < 1e-6f);
    assert(fabsf(dist_cosine(a, opposite, 2) - 2.0f) < 1e-6f);
    assert(dist_cosine(a, zero, 2) == 1.0f);
    assert(dist_cosine(zero, a, 2) == 1.0f);

    const float large[] = {FLT_MAX, -FLT_MAX};
    const float small[] = {FLT_MIN, -FLT_MIN};
    const float reversed[] = {-FLT_MIN, FLT_MIN};
    assert(fabsf(dist_cosine(large, large, 2)) < 1e-6f);
    assert(fabsf(dist_cosine(small, small, 2)) < 1e-6f);
    assert(fabsf(dist_cosine(large, small, 2)) < 1e-6f);
    assert(fabsf(dist_cosine(large, reversed, 2) - 2.0f) < 1e-6f);
}

static void test_heap_order(int is_min) {
    enum { count = 8 };
    const float distances[count] = {4.0f, 0.5f, 7.0f, 4.0f, 2.0f, 5.0f, 0.5f, 9.0f};
    const float sorted[count] = {0.5f, 0.5f, 2.0f, 4.0f, 4.0f, 5.0f, 7.0f, 9.0f};
    int seen[count] = {0};
    Heap *heap = heap_create(1, is_min);
    assert(heap != NULL);
    assert(heap_is_empty(heap));

    for (int i = 0; i < count; i++) {
        heap_push(heap, i, distances[i]);
        assert(heap_size(heap) == i + 1);
    }

    for (int i = 0; i < count; i++) {
        HeapItem top = heap_peek(heap);
        assert(top.distance == sorted[is_min ? i : count - 1 - i]);
        assert(heap_size(heap) == count - i);
        HeapItem item = heap_pop(heap);
        assert(item.distance == top.distance);
        assert(item.node_id == top.node_id);
        assert(item.node_id >= 0 && item.node_id < count);
        assert(item.distance == distances[item.node_id]);
        assert(!seen[item.node_id]);
        seen[item.node_id] = 1;
    }
    assert(heap_is_empty(heap));
    assert(heap_size(heap) == 0);
    assert(heap_peek(heap).node_id == -1 && heap_pop(heap).node_id == -1);

    heap_push(heap, 10, 1.25f);
    assert(!heap_is_empty(heap));
    HeapItem item = heap_pop(heap);
    assert(item.node_id == 10 && item.distance == 1.25f);
    assert(heap_is_empty(heap));

    for (int i = count - 1; i >= 0; --i) heap_push(heap, i, distances[i]);
    HeapItem drained[count];
    int drained_count = 0;
    heap_drain_sorted(heap, drained, &drained_count);
    assert(drained_count == count && heap_is_empty(heap));
    for (int i = 0; i < count; i++) {
        assert(drained[i].distance == sorted[i]);
        if (i > 0 && sorted[i - 1] == sorted[i]) {
            assert(drained[i - 1].node_id < drained[i].node_id);
        }
    }
    for (int i = 0; i < count; ++i) heap_push(heap, i, distances[i]);
    heap_drain_sorted(heap, heap->items, &drained_count);
    assert(drained_count == count && heap_is_empty(heap));
    for (int i = 0; i < count; ++i) {
        assert(heap->items[i].distance == drained[i].distance);
        assert(heap->items[i].node_id == drained[i].node_id);
    }
    heap_free(heap);
}

static void test_heap_retains_smallest(void) {
    const int distances[] = {8, 2, 10, 1, 6, 0, 4, 7, 3, 9, 5};
    const int keep = 4;
    Heap *results = heap_create(keep, 0);
    assert(results != NULL);

    for (size_t i = 0; i < sizeof(distances) / sizeof(distances[0]); i++) {
        heap_push(results, distances[i], (float)distances[i]);
        if (heap_size(results) > keep) {
            heap_pop(results);
        }
    }
    assert(heap_size(results) == keep);
    for (int expected = keep - 1; expected >= 0; expected--) {
        HeapItem item = heap_pop(results);
        assert(item.distance == (float)expected);
        assert(item.node_id == expected);
    }
    heap_free(results);
}

static void test_graph_node(void) {
    HNSWNode *node = node_create(0, 2, 4, 8);
    assert(node);
    node_add_neighbor(node, 0, 3);
    node_add_neighbor(node, 0, 7);
    node_add_neighbor(node, 1, 2);
    assert(node_add_neighbor(node, 0, 7));
    assert(!node_add_neighbor(node, 0, 0));
    assert(node->n_count[0] == 2 && node->n_count[1] == 1 && node->n_count[2] == 0);
    node_remove_neighbor(node, 0, 3);
    assert(node->n_count[0] == 1 && node->neighbors[0][0] == 7);
    assert(node->neighbors[1][0] == 2);
    node_free(node);
}

static void test_visited_set(void) {
    VisitedSet *visited = visited_create(5);
    assert(visited);
    assert(!visited_check(visited, 0));
    visited_mark(visited, 0);
    visited_mark(visited, 4);
    assert(visited_check(visited, 0) && visited_check(visited, 4));
    visited_reset(visited);
    assert(!visited_check(visited, 0) && !visited_check(visited, 4));
    visited_mark(visited, 2);
    assert(visited_check(visited, 2));
    assert(visited_reserve(visited, 12));
    assert(visited_check(visited, 2) && !visited_check(visited, 11));
    visited->current = UINT_MAX;
    visited_mark(visited, 11);
    visited_reset(visited);
    assert(!visited_check(visited, 11) && !visited_check(visited, 2));
    visited_mark(visited, 11);
    assert(visited_check(visited, 11));
    visited_free(visited);
}

static void test_random_levels(void) {
    int counts[4] = {0};
    utils_seed(101);
    for (int i = 0; i < 20000; i++) {
        int level = random_level(1.0f / logf(4.0f), 3);
        assert(level >= 0 && level <= 3);
        counts[level]++;
    }
    /* P(level >= 1) is 1/M. Allow for platform differences in rand(). */
    assert(counts[0] > 14000 && counts[0] < 16000);
    assert(counts[0] > counts[1] && counts[1] > counts[2]);
    assert(counts[2] > counts[3] && counts[3] > 0);
}

static void check_graph(const HNSWIndex *index) {
    int count = index->node_count;
    assert(count > 0);
    assert(index->entry_point >= 0 && index->entry_point < count);
    assert(index->nodes[index->entry_point]->level == index->max_level);
    for (int id = 0; id < count; id++) {
        const HNSWNode *node = index->nodes[id];
        assert(node->id == id && node->level >= 0 && node->level <= index->max_level);
        for (int level = 0; level <= node->level; level++) {
            int degree = node->n_count[level];
            assert(degree >= 0 && degree <= (level == 0 ? index->M0 : index->M));
            assert(degree <= node->n_cap[level]);
            for (int i = 0; i < degree; i++) {
                int neighbor = node->neighbors[level][i];
                assert(neighbor >= 0 && neighbor < count && neighbor != id);
                for (int j = 0; j < i; j++) {
                    assert(neighbor != node->neighbors[level][j]);
                }
                const HNSWNode *other = index->nodes[neighbor];
                assert(other->level >= level);
                int j = 0;
                while (j < other->n_count[level] && other->neighbors[level][j] != id) j++;
                assert(j < other->n_count[level]);
            }
        }
    }

    /* A base-layer search from the entry point must be able to reach every node. */
    unsigned char *visited = calloc((size_t)count, sizeof(*visited));
    int *queue = malloc((size_t)count * sizeof(*queue));
    assert(queue && visited);
    queue[0] = index->entry_point;
    visited[queue[0]] = 1;
    int queued = 1;
    for (int i = 0; i < queued; i++) {
        const HNSWNode *node = index->nodes[queue[i]];
        for (int j = 0; j < node->n_count[0]; j++) {
            int neighbor = node->neighbors[0][j];
            if (!visited[neighbor]) {
                visited[neighbor] = 1;
                queue[queued++] = neighbor;
            }
        }
    }
    assert(queued == count);
    free(queue);
    free(visited);
}

static void test_small_search(void) {
    const float points[][2] = {{0, 0}, {4, 0}, {0, 3}, {5, 5}};
    const float query[] = {0, 1};
    const int ids[] = {0, 2, 1, 3};
    const float distances[] = {1, 4, 17, 41};
    HNSWIndex *index = hnsw_create(2, 4, 20, 1, dist_l2_sq);
    assert(index && index->node_count == 0);
    assert(!hnsw_search(index, query, 1));
    assert(!brute_force_search(index, query, 1));
    assert(!hnsw_insert(index, points[0], 1));
    for (int i = 0; i < 4; i++) {
        float buffer[2] = {points[i][0], points[i][1]};
        assert(hnsw_insert(index, buffer, i));
        buffer[0] = -999;
        assert(index->raw_vectors[i * 2] == points[i][0]);
    }
    assert(!hnsw_insert(index, points[0], 0));
    assert(index->node_count == 4);
    assert(!hnsw_search(index, query, 0));
    assert(!hnsw_search(index, query, 5));
    assert(!brute_force_search(index, query, 5));
    KNNResult *exact = brute_force_search(index, query, 4);
    KNNResult *found = hnsw_search(index, query, 4);  /* k exceeds ef_search. */
    assert(found && exact);
    for (int i = 0; i < 4; i++) {
        assert(found[i].id == ids[i] && found[i].distance == distances[i]);
        assert(exact[i].id == ids[i] && exact[i].distance == distances[i]);
    }
    free(exact);
    free(found);
    found = hnsw_search(index, points[2], 1);
    assert(found && found[0].id == 2 && found[0].distance == 0.0f);
    free(found);
    check_graph(index);
    hnsw_free(index);
}

static void test_equal_vectors(void) {
    const float point[] = {1.0f, 1.0f};
    enum { count = 40 };
    HNSWIndex *index = hnsw_create(2, 2, 8, 2, dist_l2_sq);
    assert(index && hnsw_insert(index, point, 0));
    for (int id = 1; id < count; id++) {
        /* Use an indexed vector as input while insertion grows vector storage. */
        assert(hnsw_insert(index, index->raw_vectors, id));
    }
    check_graph(index);
    KNNResult *found = hnsw_search(index, point, count);
    assert(found);
    for (int i = 0; i < count; i++) {
        assert(found[i].id == i && found[i].distance == 0.0f);
    }
    free(found);
    hnsw_free(index);
}

static void test_recall(void) {
    enum { dim = 64, count = 1000, k = 10, queries = 100 };
    uint32_t data_seed = 2026, query_seed = 42;
    int matches = 0;
    float sample[dim];
    utils_seed(12345);
    HNSWIndex *index = hnsw_create(dim, 16, 200, 50, dist_l2_sq);
    assert(index);
    for (int i = 0; i < count; i++) {
        fill_sample(sample, dim, &data_seed);
        assert(hnsw_insert(index, sample, i));
    }
    assert(index->max_level > 0);
    check_graph(index);
    for (int q = 0; q < queries; q++) {
        fill_sample(sample, dim, &query_seed);
        KNNResult *found = hnsw_search(index, sample, k);
        KNNResult *exact = brute_force_search(index, sample, k);
        assert(found && exact);
        for (int i = 0; i < k; i++) {
            int id = found[i].id;
            assert(id >= 0 && id < count);
            assert(found[i].distance == dist_l2_sq(index->raw_vectors + id * dim, sample, dim));
            for (int j = 0; j < i; j++) assert(found[j].id != id);
            if (i > 0) {
                assert(found[i - 1].distance <= found[i].distance);
                assert(exact[i - 1].distance <= exact[i].distance);
            }
        }
        matches += count_matches(found, exact, k);
        free(found);
        free(exact);
    }
    printf("Recall@10: %.3f (100 queries, 1,000 vectors, 64 dimensions)\n",
           (double)matches / (queries * k));
    assert(matches >= 900);
    hnsw_free(index);
}

static void test_persistence(float (*metric)(const float *, const float *, int)) {
    enum { count = 80, dim = 7, k = 10 };
    char path[] = "/tmp/hnsw-test-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0 && close(fd) == 0);
    HNSWIndex *index = hnsw_create(dim, 4, 32, 12, metric);
    uint32_t seed = 193;
    float sample[dim];
    assert(index);
    assert(hnsw_save(index, path));
    HNSWIndex *loaded = hnsw_load(path);
    assert(loaded && loaded->node_count == 0 && loaded->dist_fn == metric);
    hnsw_free(loaded);

    for (int i = 0; i < count; i++) {
        fill_sample(sample, dim, &seed);
        assert(hnsw_insert(index, sample, i));
    }
    index->nodes[0]->label = strdup("first-word");
    assert(index->nodes[0]->label);
    assert(hnsw_save(index, path));
    HNSWIndex *custom = hnsw_create(dim, 4, 32, 12, dist_dot);
    assert(custom && !hnsw_save(custom, path));
    hnsw_free(custom);

    /* The previous index must still open after the unsupported save attempt. */
    loaded = hnsw_load(path);
    assert(loaded && loaded->dist_fn == metric);
    assert(loaded->dim == dim && loaded->node_count == count);
    assert(loaded->M == index->M && loaded->M0 == index->M0);
    assert(loaded->ef_construction == index->ef_construction);
    assert(loaded->ef_search == index->ef_search);
    assert(loaded->entry_point == index->entry_point && loaded->max_level == index->max_level);
    assert(memcmp(loaded->raw_vectors, index->raw_vectors, count * dim * sizeof(float)) == 0);
    for (int i = 0; i < count; i++) {
        const HNSWNode *a = index->nodes[i], *b = loaded->nodes[i];
        assert(a->id == b->id && a->level == b->level);
        if (a->label) assert(b->label && strcmp(a->label, b->label) == 0);
        else assert(!b->label);
        for (int l = 0; l <= a->level; l++) {
            assert(a->n_count[l] == b->n_count[l]);
            assert(memcmp(a->neighbors[l], b->neighbors[l], a->n_count[l] * sizeof(int)) == 0);
        }
    }
    for (int q = 0; q < 5; q++) {
        fill_sample(sample, dim, &seed);
        KNNResult *before = hnsw_search(index, sample, k);
        KNNResult *after = hnsw_search(loaded, sample, k);
        assert(before && after);
        for (int i = 0; i < k; i++) {
            assert(before[i].id == after[i].id && before[i].distance == after[i].distance);
        }
        free(before);
        free(after);
    }
    fill_sample(sample, dim, &seed);
    assert(hnsw_insert(loaded, sample, count));
    check_graph(loaded);
    hnsw_free(loaded);

    FILE *file = fopen(path, "ab");
    assert(file && fputc('!', file) != EOF && fclose(file) == 0);
    assert(!hnsw_load(path));

    /* Reject a graph with a missing reverse edge, during save or load. */
    int peer = index->nodes[0]->neighbors[0][0];
    node_remove_neighbor(index->nodes[peer], 0, 0);
    if (hnsw_save(index, path)) assert(!hnsw_load(path));
    assert(node_add_neighbor(index->nodes[peer], 0, 0));
    assert(hnsw_save(index, path));
    hnsw_free(index);

    file = fopen(path, "r+b");
    assert(file);
    assert(fseek(file, -1, SEEK_END) == 0);
    long length = ftell(file);
    assert(length > 0 && ftruncate(fileno(file), length) == 0);
    assert(fclose(file) == 0);
    assert(!hnsw_load(path));
    file = fopen(path, "wb");
    assert(file);
    assert(fputs("invalid index header", file) >= 0 && fclose(file) == 0);
    assert(!hnsw_load(path));
    assert(remove(path) == 0);
    assert(!hnsw_load(path));
}

int main(void) {
    test_vector();
    test_distances();
    test_heap_order(1);
    test_heap_order(0);
    test_heap_retains_smallest();
    test_graph_node();
    test_visited_set();
    test_random_levels();
    test_small_search();
    test_equal_vectors();
    test_recall();
    test_persistence(dist_l2_sq);
    test_persistence(dist_cosine);
    puts("All HNSW tests passed.");
    return 0;
}
