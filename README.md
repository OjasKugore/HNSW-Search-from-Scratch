# HNSW vector search in C

This CS233AI course project finds nearby vectors with a Hierarchical Navigable Small World (HNSW) graph. It includes a Mac app, a command-line demo, and a C API.

## Open Vector Search on a Mac

Double-click **VectorSearch.app** in Finder. Type a word and press **Return** to see five nearby words. You can also click an example. Select **Compare with exact search** to check the matches.

Choose **Quick**, **Balanced**, or **Thorough** to change how widely the graph searches. The **Tools** menu lets you run the engine tests or compare search speeds.

On a fresh checkout, double-click **Open Vector Search.command**. The launcher uses the Apple command-line developer tools to build the app on your Mac. It includes the 50,000-word index when the local dataset is available. Otherwise, it builds an index from the 123-word sample. After the first build, you can open the app directly or move it to another folder.

## Run the command-line demo

Use a C11 compiler, Make, and a POSIX system such as Linux or macOS. The bundled examples work offline.

```sh
make
./hnsw_demo
make test
```

The default demo indexes 1,000 generated vectors and checks 10 queries against exact search. The test suite checks the graph, distance functions, saved indexes, command-line behavior, and recall.

## Search for similar words

```sh
./hnsw_demo --vectors data/sample_vectors.txt --query king --k 5
./hnsw_demo --vectors data/sample_vectors.txt --save words.idx
./hnsw_demo --load words.idx --query paris --k 5 --verify
./hnsw_demo --load words.idx --interactive
```

Word queries use cosine distance and exclude the query word. Interactive mode reads one query per line until end of input. Add `--verify` to compare the result with an exact scan.

The sample contains 123 words selected from Stanford's GloVe 6B 50-dimensional vectors. The selection limits which words can appear in the results. The coordinate values come directly from the pretrained file. The vectors use the Public Domain Dedication and License 1.0. [GloVe project and data license](https://nlp.stanford.edu/projects/glove/)

This workspace also includes the first 50,000 vectors in `data/glove.6B.50d.first-50000.txt`. Git ignores this 21 MB local file. For a fresh checkout, extract `glove.6B.50d.txt` from the Stanford archive and supply its path. The loader reads the first 50,000 vectors by default.

```sh
./hnsw_demo --vectors data/glove.6B.50d.first-50000.txt --dim 50 --save glove.idx
./hnsw_demo --load glove.idx --query king --k 10
```

## Search numeric vectors

Supply one vector per line with whitespace between coordinates. Add `--numeric` for rows that contain only coordinates. This mode uses squared Euclidean distance. Each vector receives its zero-based row ID, excluding blank lines.

```sh
printf '0 0\n1 2\n5 7\n' > points.txt
./hnsw_demo --numeric --vectors points.txt --query-vector '1 1' --k 2
```

All rows in one file must have the same dimension and contain finite coordinates. Labeled files have one word followed by its coordinates. Use `--help` for all options.

## Measure quality and speed

```sh
make benchmark N=1000
make benchmark
```

The full benchmark builds indexes of 1,000, 10,000, and 100,000 generated vectors with 128 dimensions. It compares search widths 50, 500, and 1,000 against exact search on the same 100 queries. Recall@10 is the fraction of the ten exact neighbors that HNSW returns.

`M` controls the maximum number of upper-layer neighbors. The base layer allows twice that number. `ef_construction` sets the construction search width. `ef_search` sets the query search width. A wider search usually improves recall and takes more time. Use `--ef-search` with a saved index to change the width for a query.

One local run on arm64 macOS used Apple Clang 21 and `-O2`. The table shows mean query time for 100,000 generated vectors. Exact search averaged 3.425 ms per query.

| Search width | Recall@10 | HNSW time per query |
|---:|---:|---:|
| 50 | 24.6% | 0.182 ms |
| 500 | 84.5% | 1.637 ms |
| 1,000 | 95.2% | 2.923 ms |

## Use the C API

The public functions and types are in [src/hnsw.h](src/hnsw.h).

```c
#include "hnsw.h"
#include <stdlib.h>

int main(void) {
    float point[] = {3.0f, 4.0f};
    float query[] = {2.0f, 4.0f};
    HNSWIndex *index = hnsw_create(2, 16, 200, 50, dist_l2_sq);
    if (!index) return 1;

    KNNResult *nearest = NULL;
    if (hnsw_insert(index, point, 0)) {
        nearest = hnsw_search(index, query, 1);
    }
    int ok = nearest && nearest[0].id == 0 && nearest[0].distance == 1.0f;
    free(nearest);
    hnsw_free(index);
    return ok ? 0 : 1;
}
```

Insert IDs in sequence, starting at zero. The index copies each vector. Both search functions return exactly `k` results, ordered by distance and then ID. Supply `1 <= k <= node_count`. The caller frees the result array. A failed search returns `NULL`. Queries on one index share temporary storage. Use one thread at a time for operations on the same index.

The versioned binary format stores vectors, labels, graph edges, search settings, and the metric. It supports squared Euclidean and cosine distance. Custom distance callbacks work in memory. Saving such an index returns zero. A successful save replaces the destination after the complete write.

The [project roadmap](docs/planning.md) maps the implementation to the course modules. The [design notes](docs/METHODOLOGY.md) describe layered traversal and the two search heaps.
