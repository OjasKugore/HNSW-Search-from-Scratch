#import "../app/WordIndex.h"
#include "../src/hnsw.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    @autoreleasepool {
        NSURL *temporary = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
        NSURL *file = [temporary URLByAppendingPathComponent:NSUUID.UUID.UUIDString];
        assert(![[WordIndex alloc] initWithURL:file]);
        HNSWIndex *source = hnsw_create(2, 4, 32, 8, dist_cosine);
        assert(source);
        const char *labels[] = {"twin", "Word", "north", "west", "south", "missing link"};
        float vectors[][2] = {{1, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {-1, 1}};
        for (int i = 0; i < 6; ++i) {
            assert(hnsw_insert(source, vectors[i], i));
            source->nodes[i]->label = strdup(labels[i]);
            assert(source->nodes[i]->label);
            if (i == 0) {
                assert(hnsw_save(source, file.fileSystemRepresentation));
                WordIndex *single = [[WordIndex alloc] initWithURL:file];
                double recall = 0;
                NSArray *none = [single matchesForWord:@"twin" width:6 verify:YES recall:&recall error:nil];
                assert(single.count == 1 && none && none.count == 0 && isnan(recall));
            }
        }
        assert(hnsw_save(source, file.fileSystemRepresentation));
        WordIndex *model = [[WordIndex alloc] initWithURL:file];
        assert(model.count == 6);

        double recall = 0;
        NSError *error = nil;
        NSArray *matches = [model matchesForWord:@"  WoRD\n" width:50 verify:YES recall:&recall error:&error];
        assert(matches.count == 5 && !error && recall == 1.0);
        assert([matches[0][@"word"] isEqualToString:@"twin"]);
        assert([matches[0][@"distance"] doubleValue] == 0.0);
        assert([matches[1][@"word"] isEqualToString:@"north"]);
        assert([matches[2][@"word"] isEqualToString:@"south"]);
        assert([matches[4][@"word"] isEqualToString:@"west"]);
        assert([model matchesForWord:@"absent" width:6 verify:YES recall:&recall error:&error] == nil);
        assert(error && error.localizedDescription.length && isnan(recall));
        assert([model matchesForWord:@"word" width:6 verify:NO recall:&recall error:&error]);
        assert(!error && isnan(recall));
        assert(![model matchesForWord:@"  " width:6 verify:NO recall:NULL error:NULL]);
        assert(![model matchesForWord:@"word" width:0 verify:NO recall:NULL error:NULL]);
        hnsw_free(source);
        assert(unlink(file.fileSystemRepresentation) == 0);
    }
    return 0;
}
