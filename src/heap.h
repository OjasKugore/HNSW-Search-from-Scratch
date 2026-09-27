#ifndef HEAP_H
#define HEAP_H

typedef struct{
    int node_id;
    float distance; //dist_l2_sq
} HeapItem;

typedef struct{
    int size; //current size of the heap
    int capacity; //max capacity of the heap
    int is_min; //1 = min-heap, 0 = max-heap
    HeapItem *items;
} Heap;

Heap *heap_create(int capacity, int is_min);
void heap_push(Heap *h, int node_id, float distance);
HeapItem heap_pop(Heap *h);
HeapItem heap_peek(Heap *h);
int heap_is_empty(const Heap *h);
int heap_size(const Heap *h);
void heap_free(Heap *h);

#endif // HEAP_H
