#include "heap.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Use the node ID to break ties. NaN sorts after all numeric distances. */
static int comes_first(const Heap *h, HeapItem a, HeapItem b) {
    if (isnan(a.distance) != isnan(b.distance)) {
        return h->is_min ? isnan(b.distance) : isnan(a.distance);
    }
    if (a.distance < b.distance) {
        return h->is_min;
    }
    if (a.distance > b.distance) {
        return !h->is_min;
    }
    return h->is_min ? a.node_id < b.node_id : a.node_id > b.node_id;
}

Heap *heap_create(int capacity, int is_min) {
    if (capacity < 0 || (size_t)capacity > SIZE_MAX / sizeof(HeapItem)) {
        return NULL;
    }
    Heap *h = malloc(sizeof(*h));
    if (!h) {
        return NULL;
    }
    h->items = capacity ? malloc((size_t)capacity * sizeof(*h->items)) : NULL;
    if (capacity && !h->items) {
        free(h);
        return NULL;
    }
    h->size = 0;
    h->capacity = capacity;
    h->is_min = is_min != 0;
    return h;
}

static int grow_heap(Heap *h) {
    int capacity;
    if (h->capacity == 0) {
        capacity = 1;
    } else if (h->capacity <= INT_MAX / 2) {
        capacity = h->capacity * 2;
    } else {
        capacity = INT_MAX;
    }
    if (capacity == h->capacity || (size_t)capacity > SIZE_MAX / sizeof(*h->items)) {
        return 0;
    }
    HeapItem *items = realloc(h->items, (size_t)capacity * sizeof(*items));
    if (!items) {
        return 0;
    }
    h->items = items;
    h->capacity = capacity;
    return 1;
}

int heap_try_push(Heap *h, int node_id, float distance) {
    if (!h || (h->size == h->capacity && !grow_heap(h))) {
        return 0;
    }
    HeapItem item = {node_id, distance};
    int index = h->size++;
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (!comes_first(h, item, h->items[parent])) {
            break;
        }
        h->items[index] = h->items[parent];
        index = parent;
    }
    h->items[index] = item;
    return 1;
}

void heap_push(Heap *h, int node_id, float distance) {
    if (!heap_try_push(h, node_id, distance)) {
        fputs("Cannot insert into heap.\n", stderr);
        exit(EXIT_FAILURE);
    }
}

HeapItem heap_peek(const Heap *h) {
    if (heap_is_empty(h)) {
        return (HeapItem){-1, INFINITY};
    }
    return h->items[0];
}

HeapItem heap_pop(Heap *h) {
    HeapItem root = heap_peek(h);
    if (heap_is_empty(h)) {
        return root;
    }
    HeapItem last = h->items[--h->size];
    int index = 0;
    while (index < h->size / 2) {
        int child = 2 * index + 1;
        if (child + 1 < h->size && comes_first(h, h->items[child + 1], h->items[child])) {
            ++child;
        }
        if (!comes_first(h, h->items[child], last)) {
            break;
        }
        h->items[index] = h->items[child];
        index = child;
    }
    h->items[index] = last;
    return root;
}

void heap_replace_root(Heap *h, int node_id, float distance) {
    if (!h || h->size == 0) {
        return;
    }
    HeapItem item = {node_id, distance};
    int index = 0;
    while (index < h->size / 2) {
        int child = 2 * index + 1;
        if (child + 1 < h->size && comes_first(h, h->items[child + 1], h->items[child])) {
            ++child;
        }
        if (!comes_first(h, h->items[child], item)) {
            break;
        }
        h->items[index] = h->items[child];
        index = child;
    }
    h->items[index] = item;
}

void heap_clear(Heap *h) {
    if (h) {
        h->size = 0;
    }
}

int heap_ensure_capacity(Heap *h, int capacity) {
    if (!h || capacity < 0) {
        return 0;
    }
    if (h->capacity >= capacity) {
        return 1;
    }
    if ((size_t)capacity > SIZE_MAX / sizeof(*h->items)) {
        return 0;
    }
    HeapItem *items = realloc(h->items, (size_t)capacity * sizeof(*items));
    if (!items) {
        return 0;
    }
    h->items = items;
    h->capacity = capacity;
    return 1;
}

int heap_size(const Heap *h) {
    return h ? h->size : 0;
}

int heap_is_empty(const Heap *h) {
    return heap_size(h) == 0;
}

void heap_drain_sorted(Heap *h, HeapItem *out_array, int *out_count) {
    int count = heap_size(h);
    /* The tail slot becomes available after each pop, including in-place drains. */
    for (int i = count; i > 0; --i) {
        out_array[i - 1] = heap_pop(h);
    }
    if (h && h->is_min) {
        for (int i = 0; i < count / 2; ++i) {
            HeapItem item = out_array[i];
            out_array[i] = out_array[count - 1 - i];
            out_array[count - 1 - i] = item;
        }
    }
    *out_count = count;
}

void heap_free(Heap *h) {
    if (h) {
        free(h->items);
        free(h);
    }
}
