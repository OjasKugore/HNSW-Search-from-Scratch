#define _POSIX_C_SOURCE 200809L
#include "hnsw.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Integers and binary32 coordinates use little-endian byte order on disk. */
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 &&
               FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128, "binary32 required");
static const unsigned char signature[] = {'H', 'N', 'S', 'W', 'I', 'D', 'X', 1};
enum { METRIC_L2 = 1, METRIC_COSINE = 2 };

static int write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {value, value >> 8, value >> 16, value >> 24};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int write_index(FILE *file, const HNSWIndex *idx, int metric) {
    uint32_t header[] = {metric, idx->dim, idx->M, idx->ef_construction,
                         idx->ef_search, idx->node_count, (uint32_t)idx->entry_point + 1,
                         (uint32_t)idx->max_level + 1};
    if (fwrite(signature, 1, sizeof(signature), file) != sizeof(signature)) return 0;
    for (size_t i = 0; i < sizeof(header) / sizeof(*header); i++) {
        if (!write_u32(file, header[i])) return 0;
    }
    for (int id = 0; id < idx->node_count; id++) {
        const HNSWNode *node = idx->nodes[id];
        size_t length = node->label ? strlen(node->label) : 0;
        if (length > UINT32_MAX || !write_u32(file, node->level) ||
            !write_u32(file, (uint32_t)length)) return 0;
        if (length && fwrite(node->label, 1, length, file) != length) return 0;

        const float *vector = idx->raw_vectors + (size_t)id * idx->dim;
        for (int j = 0; j < idx->dim; j++) {
            uint32_t bits;
            memcpy(&bits, &vector[j], sizeof(bits));
            if (!write_u32(file, bits)) return 0;
        }
        for (int layer = 0; layer <= node->level; layer++) {
            if (!write_u32(file, node->n_count[layer])) return 0;
            for (int j = 0; j < node->n_count[layer]; j++) {
                if (!write_u32(file, node->neighbors[layer][j])) return 0;
            }
        }
    }
    return 1;
}

int hnsw_save(const HNSWIndex *idx, const char *path) {
    if (!idx || !path || !*path) return 0;
    int metric;
    if (idx->dist_fn == dist_l2_sq) metric = METRIC_L2;
    else if (idx->dist_fn == dist_cosine) metric = METRIC_COSINE;
    else return 0;

    const char suffix[] = ".XXXXXX";
    size_t size = strlen(path);
    if (size > SIZE_MAX - sizeof(suffix)) return 0;
    char *temporary = malloc(size + sizeof(suffix));
    if (!temporary) return 0;
    memcpy(temporary, path, size);
    memcpy(temporary + size, suffix, sizeof(suffix));
    int fd = mkstemp(temporary);
    int ok = 0;
    if (fd >= 0) {
        FILE *file = fdopen(fd, "wb");
        if (file) {
            ok = write_index(file, idx, metric);
            if (fclose(file) != 0) ok = 0;
        } else {
            close(fd);
        }
        if (ok) ok = rename(temporary, path) == 0;
        if (!ok) unlink(temporary);
    }
    free(temporary);
    return ok;
}

typedef struct {
    FILE *file;
    uint64_t remaining;
} Reader;

static int read_bytes(Reader *in, void *out, size_t count) {
    if (count > in->remaining || fread(out, 1, count, in->file) != count) return 0;
    in->remaining -= count;
    return 1;
}

static int read_u32(Reader *in, uint32_t *value) {
    unsigned char bytes[4];
    if (!read_bytes(in, bytes, sizeof(bytes))) return 0;
    *value = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
             (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
    return 1;
}

static int read_node(Reader *in, HNSWIndex *idx, int id, int total) {
    uint32_t level, length;
    if (!read_u32(in, &level) || level > HNSW_MAX_LEVEL ||
        !read_u32(in, &length) || length > in->remaining ||
        (uint64_t)length + 1 > SIZE_MAX) return 0;
    HNSWNode *node = node_create(id, (int)level, idx->M, idx->M0);
    if (!node) return 0;
    idx->nodes[id] = node;
    idx->node_count++; /* hnsw_free owns this row, including a partial read. */
    if (length) {
        node->label = malloc((size_t)length + 1);
        if (!node->label || !read_bytes(in, node->label, length) ||
            memchr(node->label, 0, length)) return 0;
        node->label[length] = '\0';
    }
    float *vector = idx->raw_vectors + (size_t)id * idx->dim;
    for (int j = 0; j < idx->dim; j++) {
        uint32_t bits;
        if (!read_u32(in, &bits)) return 0;
        memcpy(vector + j, &bits, sizeof(bits));
        if (!isfinite(vector[j])) return 0;
    }
    for (int layer = 0; layer <= node->level; layer++) {
        uint32_t count;
        unsigned int limit = layer == 0 ? idx->M0 : idx->M;
        if (!read_u32(in, &count) || count > limit || count >= (unsigned int)total)
            return 0;
        visited_reset(idx->visited);
        for (uint32_t j = 0; j < count; j++) {
            uint32_t neighbor;
            if (!read_u32(in, &neighbor) || neighbor >= (uint32_t)total ||
                neighbor == (uint32_t)id || visited_check(idx->visited, (int)neighbor))
                return 0;
            visited_mark(idx->visited, (int)neighbor);
            node->neighbors[layer][node->n_count[layer]++] = (int)neighbor;
        }
    }
    return 1;
}

static int valid_graph(const HNSWIndex *idx) {
    if (!idx->node_count) return 1;
    if (idx->nodes[idx->entry_point]->level != idx->max_level) return 0;
    for (int id = 0; id < idx->node_count; id++) {
        HNSWNode *node = idx->nodes[id];
        if (node->level > idx->max_level) return 0;
        int has_predecessor = id == 0;
        for (int layer = 0; layer <= node->level; layer++) {
            for (int j = 0; j < node->n_count[layer]; j++) {
                HNSWNode *peer = idx->nodes[node->neighbors[layer][j]];
                if (layer == 0 && peer->id == id - 1) has_predecessor = 1;
                if (peer->level < layer) return 0;
                int r = 0;
                while (r < peer->n_count[layer] && peer->neighbors[layer][r] != id) r++;
                if (r == peer->n_count[layer]) return 0;
            }
        }
        /* Consecutive IDs must stay linked after future insertions. */
        if (!has_predecessor) return 0;
    }
    return 1;
}

static HNSWIndex *read_index(Reader *in) {
    unsigned char magic[sizeof(signature)];
    if (!read_bytes(in, magic, sizeof(magic)) || memcmp(magic, signature, sizeof(magic)))
        return NULL;
    enum { METRIC, DIM, M, CONSTRUCTION, SEARCH, COUNT, ENTRY, LEVEL, HEADER_FIELDS };
    uint32_t header[HEADER_FIELDS];
    for (int i = 0; i < HEADER_FIELDS; i++) {
        if (!read_u32(in, &header[i]) || header[i] > INT_MAX) return NULL;
    }
    int total = (int)header[COUNT];
    if (header[ENTRY] > header[COUNT] || header[LEVEL] > HNSW_MAX_LEVEL + 1 ||
        (total != 0) != (header[ENTRY] != 0) || (total != 0) != (header[LEVEL] != 0))
        return NULL;
    /* A row needs its level, label length, base degree, and all coordinates. */
    uint64_t min_row = 12 + UINT64_C(4) * header[DIM];
    if ((uint64_t)total > in->remaining / min_row) return NULL;

    DistanceFn metric;
    if (header[METRIC] == METRIC_L2) metric = dist_l2_sq;
    else if (header[METRIC] == METRIC_COSINE) metric = dist_cosine;
    else return NULL;
    HNSWIndex *idx = hnsw_create(header[DIM], header[M], header[CONSTRUCTION], header[SEARCH], metric);
    if (!idx) return NULL;
    idx->entry_point = (int)header[ENTRY] - 1;
    idx->max_level = (int)header[LEVEL] - 1;
    if (total && !hnsw_reserve(idx, total)) goto invalid;
    for (int id = 0; id < total; id++) {
        if (!read_node(in, idx, id, total)) goto invalid;
    }
    if (in->remaining || !valid_graph(idx)) goto invalid;
    return idx;

invalid:
    hnsw_free(idx);
    return NULL;
}

HNSWIndex *hnsw_load(const char *path) {
    if (!path || !*path) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    struct stat st;
    HNSWIndex *idx = NULL;
    if (fstat(fileno(file), &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0) {
        Reader in = {file, (uint64_t)st.st_size};
        idx = read_index(&in);
    }
    fclose(file);
    return idx;
}
