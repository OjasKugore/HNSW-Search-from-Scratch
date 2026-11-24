#include "utils.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int visited_reserve(VisitedSet *vs, int capacity) {
    if (!vs || capacity < 0 || (size_t)capacity > SIZE_MAX / sizeof(*vs->gen))
        return 0;
    if (capacity <= vs->capacity) return 1;

    unsigned int *gen = realloc(vs->gen, (size_t)capacity * sizeof(*gen));
    if (!gen) return 0;
    memset(gen + vs->capacity, 0, (size_t)(capacity - vs->capacity) * sizeof(*gen));
    vs->gen = gen;
    vs->capacity = capacity;
    return 1;
}

VisitedSet *visited_create(int capacity) {
    VisitedSet *vs = calloc(1, sizeof(*vs));
    if (!vs || !visited_reserve(vs, capacity)) {
        free(vs);
        return NULL;
    }
    vs->current = 1;
    return vs;
}

void visited_mark(VisitedSet *vs, int id) {
    if (vs && id >= 0 && id < vs->capacity) vs->gen[id] = vs->current;
}

int visited_check(const VisitedSet *vs, int id) {
    return vs && id >= 0 && id < vs->capacity && vs->gen[id] == vs->current;
}

void visited_reset(VisitedSet *vs) {
    if (vs && ++vs->current == 0) {
        if (vs->capacity) memset(vs->gen, 0, (size_t)vs->capacity * sizeof(*vs->gen));
        vs->current = 1;
    }
}

void visited_free(VisitedSet *vs) {
    if (vs) free(vs->gen);
    free(vs);
}

void utils_seed(unsigned int seed) {
    srand(seed);
}

int random_level(float ml, int max_cap) {
    if (!(ml > 0.0f) || max_cap <= 0) return 0;
    double uniform = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double level = -log(uniform) * ml;
    return level < max_cap ? (int)level : max_cap;
}
