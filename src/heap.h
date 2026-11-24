#ifndef HEAP_H
#define HEAP_H

enum { HEAP_MAX = 0, HEAP_MIN = 1 };

typedef struct {
    int node_id;
    float distance;
} HeapItem;

typedef struct {
    int size;
    int capacity;
    int is_min;
    HeapItem *items;
} Heap;

/* Capacity is an initial reservation and may be zero. Allocation failure returns NULL. */
Heap *heap_create(int capacity, int is_min);
/* Return zero if insertion fails. The heap retains all its original items. */
int heap_try_push(Heap *h, int node_id, float distance);
/* Insert one item or terminate the program on failure. */
void heap_push(Heap *h, int node_id, float distance);
/* An empty heap returns an item with node_id -1 and infinite distance. */
HeapItem heap_pop(Heap *h);
HeapItem heap_peek(const Heap *h);
/* Replace the root item and sift down in a single pass. */
void heap_replace_root(Heap *h, int node_id, float distance);
void heap_clear(Heap *h);
int heap_ensure_capacity(Heap *h, int capacity);
int heap_is_empty(const Heap *h);
int heap_size(const Heap *h);
/* Write heap_size(h) items sorted by distance then ID. Output may equal h->items. */
void heap_drain_sorted(Heap *h, HeapItem *out_array, int *out_count);
void heap_free(Heap *h);

#endif // HEAP_H
