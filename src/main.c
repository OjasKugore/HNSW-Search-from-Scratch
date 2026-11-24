#define _POSIX_C_SOURCE 200809L
#include "hnsw.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
    const char *vectors, *load, *save, *word, *coordinates;
    int dim, M, ef_construction, ef_search, limit, k, queries;
    int numeric, interactive, verify;
} Options;

static void usage(FILE *out) {
    fputs(
        "Usage: hnsw_demo [options]\n"
        "With no arguments, benchmark 1000 random 128-dimensional vectors and 10 queries.\n"
        "  --vectors FILE        Read whitespace-delimited word embeddings (cosine distance).\n"
        "  --numeric             Read coordinate-only rows (squared Euclidean distance).\n"
        "  --load FILE           Open a saved index, including any word labels.\n"
        "  --save FILE           Save the index after building or loading it.\n"
        "  --dim N               Require N coordinates per row (default: infer from file).\n"
        "  --limit N             Number of vectors to generate or read (file default: 50000).\n"
        "  --M N                 Upper-layer neighbor limit, 2 to 256 (default: 16).\n"
        "  --ef-construction N   Construction search width (default: 200).\n"
        "  --ef-search N         Query search width (default: 50 or saved setting).\n"
        "  --k N                 Maximum number of neighbors to show (default: 10).\n"
        "  --query WORD          Find neighbors of an indexed word and exclude the word itself.\n"
        "  --query-vector 'X Y'  Find neighbors of a vector. Supply all coordinates.\n"
        "  --interactive         Read one query per line until EOF.\n"
        "  --verify              Compare each requested result list with exact search.\n"
        "  --queries N           Benchmark N repeatable random queries against exact search.\n"
        "  --help                Show this message.\n", out);
}

static int positive_integer(const char *text, int *out) {
    char *end;
    errno = 0;
    long n = strtol(text, &end, 10);
    if (end == text || *end || errno || n < 1 || n > INT_MAX) {
        return 0;
    }
    *out = (int)n;
    return 1;
}

static int parse_options(Options *o, int argc, char **argv) {
    struct { const char *name; int *value; } numbers[] = {
        {"--dim", &o->dim}, {"--M", &o->M}, {"--ef-construction", &o->ef_construction},
        {"--ef-search", &o->ef_search}, {"--limit", &o->limit}, {"--k", &o->k},
        {"--queries", &o->queries}
    };
    struct { const char *name; const char **value; } strings[] = {
        {"--vectors", &o->vectors}, {"--load", &o->load}, {"--save", &o->save},
        {"--query", &o->word}, {"--query-vector", &o->coordinates}
    };
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--numeric")) {
            o->numeric = 1;
        } else if (!strcmp(arg, "--interactive")) {
            o->interactive = 1;
        } else if (!strcmp(arg, "--verify")) {
            o->verify = 1;
        } else if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            exit(EXIT_SUCCESS);
        } else {
            int known = 0;
            for (size_t j = 0; j < sizeof(numbers) / sizeof(numbers[0]); ++j) {
                if (!strcmp(arg, numbers[j].name)) {
                    if (++i == argc || !positive_integer(argv[i], numbers[j].value)) {
                        fprintf(stderr, "%s requires a positive integer.\n", arg);
                        return 0;
                    }
                    known = 1;
                    break;
                }
            }
            for (size_t j = 0; !known && j < sizeof(strings) / sizeof(strings[0]); ++j) {
                if (!strcmp(arg, strings[j].name)) {
                    if (++i == argc || !*argv[i]) {
                        fprintf(stderr, "%s requires a value.\n", arg);
                        return 0;
                    }
                    *strings[j].value = argv[i];
                    known = 1;
                }
            }
            if (!known) {
                fprintf(stderr, "Unknown option: %s\n", arg);
                return 0;
            }
        }
    }
    if (o->load && (o->vectors || o->numeric || o->dim || o->M || o->ef_construction || o->limit)) {
        fputs("--load cannot be combined with index construction options.\n", stderr);
        return 0;
    }
    if (o->numeric && !o->vectors) {
        fputs("--numeric requires --vectors.\n", stderr);
        return 0;
    }
    if (o->vectors && o->save) {
        struct stat input, output;
        if (stat(o->vectors, &input) == 0 && stat(o->save, &output) == 0 &&
            input.st_dev == output.st_dev && input.st_ino == output.st_ino) {
            fputs("The output index would replace the input vectors. Select another --save path.\n", stderr);
            return 0;
        }
    }
    if (o->M && (o->M < 2 || o->M > HNSW_MAX_M)) {
        fprintf(stderr, "--M must be between 2 and %d.\n", HNSW_MAX_M);
        return 0;
    }
    return 1;
}

static char *skip_space(char *p) {
    while (isspace((unsigned char)*p)) {
        ++p;
    }
    return p;
}

static int count_fields(char *row) {
    int n = 0;
    for (char *p = skip_space(row); *p; p = skip_space(p)) {
        if (n == INT_MAX) {
            return 0;
        }
        ++n;
        while (*p && !isspace((unsigned char)*p)) {
            ++p;
        }
    }
    return n;
}

static int read_coordinates(const char *text, float *values, int dim) {
    for (int i = 0; i < dim; ++i) {
        char *end;
        float value = strtof(text, &end);
        if (end == text || (*end && !isspace((unsigned char)*end)) || !isfinite(value)) {
            return 0;
        }
        values[i] = value;
        text = end;
    }
    while (isspace((unsigned char)*text)) {
        ++text;
    }
    return !*text;
}

static float *vector_buffer(int dim) {
    if (dim <= 0 || (size_t)dim > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    return malloc((size_t)dim * sizeof(float));
}

static HNSWIndex *create_index(const Options *o, int dim, DistanceFn metric) {
    HNSWIndex *idx = hnsw_create(dim, o->M ? o->M : 16, o->ef_construction ? o->ef_construction : 200,
                                 o->ef_search ? o->ef_search : 50, metric);
    if (!idx) {
        fputs("Cannot create an index with the requested settings.\n", stderr);
    }
    return idx;
}

static HNSWIndex *read_vectors(const Options *o) {
    FILE *fp = fopen(o->vectors, "r");
    if (!fp) {
        perror(o->vectors);
        return NULL;
    }
    HNSWIndex *idx = NULL;
    float *values = NULL;
    char *line = NULL;
    size_t capacity = 0, line_number = 0;
    int dim = o->dim, limit = o->limit ? o->limit : 50000;
    while (!idx || idx->node_count < limit) {
        ssize_t length = getline(&line, &capacity, fp);
        if (length < 0) {
            if (!feof(fp)) {
                perror(o->vectors);
                goto fail;
            }
            break;
        }
        ++line_number;
        char *row = skip_space(line);
        if (memchr(line, '\0', (size_t)length)) {
            fprintf(stderr, "%s:%zu: unexpected null byte.\n", o->vectors, line_number);
            goto fail;
        }
        if (!*row) {
            continue;
        }
        char *word = NULL;
        if (!o->numeric) {
            word = row;
            while (*row && !isspace((unsigned char)*row)) {
                ++row;
            }
            if (*row) {
                *row++ = '\0';
            }
        }
        if (!idx) {
            dim = dim ? dim : count_fields(row);
            values = vector_buffer(dim);
            if (!values) {
                fprintf(stderr, "%s:%zu: missing or unsupported vector dimensions.\n", o->vectors, line_number);
                goto fail;
            }
            idx = create_index(o, dim, o->numeric ? dist_l2_sq : dist_cosine);
            if (!idx) {
                goto fail;
            }
        }
        if (!read_coordinates(row, values, dim)) {
            fprintf(stderr, "%s:%zu: expected %d finite coordinates.\n", o->vectors, line_number, dim);
            goto fail;
        }
        int id = idx->node_count;
        if (!hnsw_insert(idx, values, id)) {
            fprintf(stderr, "%s:%zu: cannot insert vector.\n", o->vectors, line_number);
            goto fail;
        }
        if (word && !(idx->nodes[id]->label = strdup(word))) {
            fputs("Cannot allocate a word label.\n", stderr);
            goto fail;
        }
    }
    if (!idx) {
        fprintf(stderr, "%s: no vectors found.\n", o->vectors);
    }
    goto finish;
fail:
    hnsw_free(idx);
    idx = NULL;
finish:
    free(values);
    free(line);
    fclose(fp);
    return idx;
}

/* Keep query and dataset generation independent of random graph levels. */
static void random_vector(uint32_t *state, float *values, int dim) {
    for (int i = 0; i < dim; ++i) {
        *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
        values[i] = (float)(*state >> 8) / 16777216.0f;
    }
}

static HNSWIndex *generate_vectors(const Options *o) {
    int dim = o->dim ? o->dim : 128;
    int count = o->limit ? o->limit : 1000;
    HNSWIndex *idx = create_index(o, dim, dist_l2_sq);
    float *values = vector_buffer(dim);
    if (!idx || !values || !hnsw_reserve(idx, count)) {
        fputs("Cannot allocate the random dataset.\n", stderr);
        hnsw_free(idx);
        free(values);
        return NULL;
    }
    uint32_t state = 42;
    for (int id = 0; id < count; ++id) {
        random_vector(&state, values, dim);
        if (!hnsw_insert(idx, values, id)) {
            fprintf(stderr, "Cannot insert vector %d.\n", id);
            hnsw_free(idx);
            idx = NULL;
            break;
        }
    }
    free(values);
    return idx;
}

typedef struct {
    double approx_seconds, exact_seconds, recall;
} Measurements;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Compact the result in place. Tied vectors can appear before the query word. */
static void exclude_id(KNNResult *results, int count, int k, int exclude) {
    if (exclude < 0) {
        return;
    }
    int written = 0;
    for (int i = 0; i < count && written < k; ++i) {
        if (results[i].id != exclude) {
            results[written++] = results[i];
        }
    }
}

static int search_once(HNSWIndex *idx, const float *query, int k, int exclude,
                       int show, Measurements *measurements) {
    int count = k + (exclude >= 0);
    double start = measurements ? now() : 0.0;
    KNNResult *found = hnsw_search(idx, query, count);
    if (!found) {
        fputs("Cannot complete the search.\n", stderr);
        return 0;
    }
    if (measurements) {
        measurements->approx_seconds += now() - start;
    }
    exclude_id(found, count, k, exclude);
    if (show) {
        for (int i = 0; i < k; ++i) {
            int id = found[i].id;
            printf("%2d.  ", i + 1);
            if (idx->nodes[id]->label) {
                printf("%s", idx->nodes[id]->label);
            } else {
                printf("vector %d", id);
            }
            printf("  %.6f\n", found[i].distance);
        }
    }
    if (measurements) {
        start = now();
        KNNResult *exact = brute_force_search(idx, query, count);
        measurements->exact_seconds += now() - start;
        unsigned char *expected = calloc((size_t)idx->node_count, 1);
        if (!exact || !expected) {
            fputs("Cannot verify the search.\n", stderr);
            free(found);
            free(exact);
            free(expected);
            return 0;
        }
        exclude_id(exact, count, k, exclude);
        for (int i = 0; i < k; ++i) {
            expected[exact[i].id] = 1;
        }
        int hits = 0;
        for (int i = 0; i < k; ++i) {
            hits += expected[found[i].id];
        }
        measurements->recall += (double)hits / k;
        free(expected);
        free(exact);
    }
    free(found);
    return 1;
}

static void print_measurements(const Measurements *m, int k, int queries) {
    printf("Recall@%d: %.4f (%d %s)\n", k, m->recall / queries, queries, queries == 1 ? "query" : "queries");
    printf("Mean query time: HNSW %.3f us, brute force %.3f us\n",
           m->approx_seconds * 1e6 / queries, m->exact_seconds * 1e6 / queries);
}

static int benchmark(HNSWIndex *idx, int k, int queries) {
    float *q = vector_buffer(idx->dim);
    if (!q) {
        fputs("Cannot allocate a query vector.\n", stderr);
        return 0;
    }
    uint32_t state = 12345;
    Measurements m = {0};
    puts("Neighbors of the first random query (distance):");
    for (int i = 0; i < queries; ++i) {
        random_vector(&state, q, idx->dim);
        if (!search_once(idx, q, k, -1, i == 0, &m)) {
            free(q);
            return 0;
        }
    }
    free(q);
    print_measurements(&m, k, queries);
    return 1;
}

/* A positive return reports success, zero reports bad input, and -1 reports a search failure. */
static int run_query(HNSWIndex *idx, const char *input, int k, int word, int verify) {
    int exclude = -1;
    float *values = NULL;
    const float *q;
    if (word) {
        for (int i = 0; i < idx->node_count; ++i) {
            if (idx->nodes[i]->label && !strcmp(input, idx->nodes[i]->label)) {
                exclude = i;
                break;
            }
        }
        if (exclude < 0) {
            fprintf(stderr, "Word absent from this index: %s\n", input);
            return 0;
        }
        if (k >= idx->node_count) {
            k = idx->node_count - 1;
        }
        if (!k) {
            puts("This index contains only the query word.");
            return 1;
        }
        q = idx->raw_vectors + (size_t)exclude * idx->dim;
    } else {
        values = vector_buffer(idx->dim);
        if (!values) {
            fputs("Cannot allocate a query vector.\n", stderr);
            return -1;
        }
        if (!read_coordinates(input, values, idx->dim)) {
            fprintf(stderr, "Query requires %d finite coordinates.\n", idx->dim);
            free(values);
            return 0;
        }
        q = values;
    }
    printf("Neighbors of %s (distance):\n", input);
    Measurements m = {0};
    int ok = search_once(idx, q, k, exclude, 1, verify ? &m : NULL);
    free(values);
    if (ok && verify) {
        print_measurements(&m, k, 1);
    }
    return ok ? 1 : -1;
}

static int interactive(HNSWIndex *idx, int k, int verify, int words) {
    char *line = NULL;
    size_t capacity = 0;
    printf("Enter one %s per line. End input with EOF.\n", words ? "word" : "vector");
    for (;;) {
        fflush(stdout);
        ssize_t length = getline(&line, &capacity, stdin);
        if (length < 0) {
            break;
        }
        while (length && isspace((unsigned char)line[length - 1])) {
            line[--length] = '\0';
        }
        char *input = skip_space(line);
        if (*input && run_query(idx, input, k, words, verify) < 0) {
            free(line);
            return 0;
        }
    }
    free(line);
    return feof(stdin) != 0;
}

int main(int argc, char **argv) {
    Options o = {.k = 10};
    if (!parse_options(&o, argc, argv)) {
        return EXIT_FAILURE;
    }
    utils_seed(42);
    HNSWIndex *idx;
    if (o.load) {
        idx = hnsw_load(o.load);
        if (!idx) {
            fprintf(stderr, "Cannot load index: %s\n", o.load);
            return EXIT_FAILURE;
        }
        if (o.ef_search) {
            idx->ef_search = o.ef_search;
        }
    } else {
        idx = o.vectors ? read_vectors(&o) : generate_vectors(&o);
    }
    if (!idx) {
        return EXIT_FAILURE;
    }

    int ok = 1;
    hnsw_print_stats(idx);
    printf("Metric: %s\n", idx->dist_fn == dist_cosine ? "cosine distance" : "squared Euclidean distance");
    if (o.save) {
        ok = hnsw_save(idx, o.save);
        if (!ok) {
            fprintf(stderr, "Cannot save index: %s\n", o.save);
        }
    }
    int has_query = o.word || o.coordinates || o.queries || o.interactive;
    if (!ok || (!has_query && o.save)) {
        hnsw_free(idx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (!idx->node_count) {
        fputs("This index contains no vectors.\n", stderr);
        hnsw_free(idx);
        return EXIT_FAILURE;
    }

    int k = o.k < idx->node_count ? o.k : idx->node_count;
    int words = 0;
    if (o.interactive || !has_query) {
        for (int i = 0; i < idx->node_count && !words; i++) {
            words = idx->nodes[i]->label != NULL;
        }
    }
    if (ok && o.word) {
        ok = run_query(idx, o.word, k, 1, o.verify) > 0;
    }
    if (ok && o.coordinates) {
        ok = run_query(idx, o.coordinates, k, 0, o.verify) > 0;
    }
    if (ok && (o.queries || (!has_query && !words))) {
        ok = benchmark(idx, k, o.queries ? o.queries : 10);
    }
    if (ok && (o.interactive || (!has_query && words))) {
        ok = interactive(idx, k, o.verify, words);
    }
    hnsw_free(idx);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
