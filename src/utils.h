#ifndef UTILS_H
#define UTILS_H

typedef struct {
    unsigned int *gen;
    unsigned int current;
    int capacity;
} VisitedSet;

VisitedSet *visited_create(int capacity);
int visited_reserve(VisitedSet *vs, int capacity);
void visited_mark(VisitedSet *vs, int id);
int visited_check(const VisitedSet *vs, int id);
void visited_reset(VisitedSet *vs);
void visited_free(VisitedSet *vs);

void utils_seed(unsigned int seed);
int random_level(float ml, int max_cap);

#endif
