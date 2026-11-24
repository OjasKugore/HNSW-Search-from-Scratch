#import "WordIndex.h"
#include "../src/hnsw.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>

static NSString *wordKey(NSString *word) {
    return [[word stringByTrimmingCharactersInSet:
            NSCharacterSet.whitespaceAndNewlineCharacterSet] lowercaseString];
}

static NSArray *searchError(NSError **error, NSInteger code, NSString *message) {
    if (error) {
        *error = [NSError errorWithDomain:@"WordIndex" code:code
                                 userInfo:@{NSLocalizedDescriptionKey: message}];
    }
    return nil;
}

/* Search asks for one extra entry. Skip the query ID wherever it ranks.
   If approximate search missed the query ID, discard the extra entry. */
static void excludeQuery(KNNResult *results, int limit, int queryID) {
    int kept = 0;
    for (int i = 0; kept < limit; ++i) {
        if (results[i].id != queryID) results[kept++] = results[i];
    }
}

@implementation WordIndex {
    HNSWIndex *_index;
    NSDictionary<NSString *, NSNumber *> *_wordIDs;
}

- (nullable instancetype)initWithURL:(NSURL *)url {
    self = [super init];
    if (!self || !url.isFileURL) return nil;
    _index = hnsw_load(url.fileSystemRepresentation);
    if (!_index || _index->node_count == 0 || _index->dist_fn != dist_cosine) return nil;

    NSMutableDictionary *wordIDs = [NSMutableDictionary dictionaryWithCapacity:_index->node_count];
    for (int i = 0; i < _index->node_count; ++i) {
        const char *label = _index->nodes[i]->label;
        if (!label) return nil;
        NSString *key = wordKey([NSString stringWithUTF8String:label]);
        if (!key.length) return nil;
        if (!wordIDs[key]) wordIDs[key] = @(i);
    }
    _wordIDs = wordIDs;
    return self;
}

- (void)dealloc {
    hnsw_free(_index);
}

- (NSInteger)count {
    return _index->node_count;
}

- (nullable NSArray<NSDictionary *> *)matchesForWord:(NSString *)word
                                               width:(NSInteger)width
                                              verify:(BOOL)verify
                                              recall:(double *)recall
                                               error:(NSError **)error {
    if (recall) *recall = NAN;
    if (error) *error = nil;
    NSString *key = wordKey(word);
    if (!key.length) return searchError(error, 1, @"Enter a word to search.");
    if (width < 1 || width > INT_MAX) return searchError(error, 2, @"Select a valid search width.");
    NSNumber *identifier = _wordIDs[key];
    if (!identifier) {
        NSString *message = [NSString stringWithFormat:@"“%@” is missing from this collection. Try another word.", key];
        return searchError(error, 3, message);
    }

    int queryID = identifier.intValue;
    int limit = MIN(5, _index->node_count - 1);
    if (limit == 0) return @[];
    const float *query = _index->raw_vectors + (size_t)queryID * _index->dim;
    _index->ef_search = (int)width;
    KNNResult *found = hnsw_search(_index, query, limit + 1);
    if (!found) return searchError(error, 4, @"The word search failed. Try again.");
    excludeQuery(found, limit, queryID);

    if (verify) {
        KNNResult *exact = brute_force_search(_index, query, limit + 1);
        if (!exact) {
            free(found);
            return searchError(error, 5, @"The exact search failed. Try again.");
        }
        excludeQuery(exact, limit, queryID);
        int hits = 0;
        for (int i = 0; i < limit; ++i) {
            for (int j = 0; j < limit; ++j) {
                if (found[i].id == exact[j].id) {
                    ++hits;
                    break;
                }
            }
        }
        free(exact);
        if (recall) *recall = (double)hits / limit;
    }

    NSMutableArray *matches = [NSMutableArray arrayWithCapacity:limit];
    for (int i = 0; i < limit; ++i) {
        NSString *label = [NSString stringWithUTF8String:_index->nodes[found[i].id]->label];
        [matches addObject:@{@"word": label, @"distance": @(found[i].distance)}];
    }
    free(found);
    return matches;
}

@end
