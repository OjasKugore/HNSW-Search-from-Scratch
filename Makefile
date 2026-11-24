CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic
LDLIBS += -lm

CORE = src/vector.c src/heap.c src/utils.c src/graph.c src/hnsw.c src/persistence.c
HEADERS = $(wildcard src/*.h)

.PHONY: all test benchmark clean
all: hnsw_demo

hnsw_demo: src/main.c $(CORE) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ src/main.c $(CORE) $(LDLIBS)

test_hnsw: tests/test_hnsw.c tests/workload.h $(CORE) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ tests/test_hnsw.c $(CORE) $(LDLIBS)

test: test_hnsw hnsw_demo
	./test_hnsw
	sh tests/test_cli.sh

hnsw_bench: tests/benchmark.c tests/workload.h $(CORE) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ tests/benchmark.c $(CORE) $(LDLIBS)

benchmark: hnsw_bench
	./hnsw_bench $(N)

# Desktop app for macOS. Keep the C library, CLI and benchmark portable.
.PHONY: app
ifeq ($(shell uname -s),Darwin)
BUNDLE = VectorSearch.app/Contents
APP_CACHE = .build/vector-search
GLOVE = data/glove.6B.50d.first-50000.txt
APP_DATA = $(if $(wildcard $(GLOVE)),$(GLOVE),data/sample_vectors.txt)
APP_INDEX = $(APP_CACHE)/$(notdir $(APP_DATA)).idx
APP_SOURCES = app/main.m app/WordIndex.m app/SearchTools.m
APP_FLAGS = -fobjc-arc -mmacosx-version-min=12.0 -Isrc

app: $(BUNDLE)/MacOS/VectorSearch $(BUNDLE)/Info.plist $(APP_INDEX) $(BUNDLE)/Helpers/test_hnsw $(BUNDLE)/Helpers/hnsw_bench
	@mkdir -p "$(BUNDLE)/Resources"
	@if ! cmp -s "$(APP_INDEX)" "$(BUNDLE)/Resources/words.idx"; then \
	    cp "$(APP_INDEX)" "$(BUNDLE)/Resources/words.idx.new" && \
	    mv "$(BUNDLE)/Resources/words.idx.new" "$(BUNDLE)/Resources/words.idx"; \
	fi
	@if [ ! -f "$(BUNDLE)/_CodeSignature/CodeResources" ] || ! codesign --verify --deep --strict VectorSearch.app 2>/dev/null; then \
	    codesign --force --sign - VectorSearch.app; \
	fi

$(BUNDLE)/MacOS/VectorSearch: $(APP_SOURCES) app/WordIndex.h app/SearchTools.h $(CORE) $(HEADERS)
	@mkdir -p "$(@D)"
	$(CC) $(CPPFLAGS) $(CFLAGS) $(APP_FLAGS) $(LDFLAGS) -o "$@" $(APP_SOURCES) $(CORE) -framework Cocoa $(LDLIBS)

$(BUNDLE)/Info.plist: app/Info.plist
	@mkdir -p "$(@D)"
	cp "$<" "$@"

$(APP_INDEX): $(APP_DATA) hnsw_demo
	@mkdir -p "$(@D)"
	@echo "Preparing the offline word index. This can take a minute."
	@./hnsw_demo --vectors "$(APP_DATA)" --save "$@" >/dev/null

$(BUNDLE)/Helpers/test_hnsw: tests/test_hnsw.c tests/workload.h $(CORE) $(HEADERS) Makefile
	@mkdir -p "$(@D)"
	$(CC) $(CPPFLAGS) $(CFLAGS) -mmacosx-version-min=12.0 $(LDFLAGS) -o "$@" "$<" $(CORE) $(LDLIBS)
	codesign --force --sign - "$@"

$(BUNDLE)/Helpers/hnsw_bench: tests/benchmark.c tests/workload.h $(CORE) $(HEADERS) Makefile
	@mkdir -p "$(@D)"
	$(CC) $(CPPFLAGS) $(CFLAGS) -mmacosx-version-min=12.0 $(LDFLAGS) -o "$@" "$<" $(CORE) $(LDLIBS)
	codesign --force --sign - "$@"

$(APP_CACHE)/test_word_index: tests/test_word_index.m app/WordIndex.m app/WordIndex.h $(CORE) $(HEADERS)
	@mkdir -p "$(@D)"
	$(CC) $(CPPFLAGS) $(CFLAGS) $(APP_FLAGS) -Iapp $(LDFLAGS) -o "$@" tests/test_word_index.m app/WordIndex.m $(CORE) -framework Foundation $(LDLIBS)

.PHONY: test-app
test-app: $(APP_CACHE)/test_word_index
	./$(APP_CACHE)/test_word_index

test: test-app
else
app:
	@echo "The desktop app requires macOS. Use make to build the command-line program." >&2
	@exit 1
endif

clean:
	$(RM) hnsw_demo test_hnsw hnsw_bench
	$(RM) -r hnsw_demo.dSYM test_hnsw.dSYM hnsw_bench.dSYM VectorSearch.app VectorSearch.app.dSYM .build/vector-search
