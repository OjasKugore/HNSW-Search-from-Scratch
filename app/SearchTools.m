#import "SearchTools.h"

#include <math.h>
#include <stdio.h>

static NSArray<NSArray<NSString *> *> *BenchmarkRows(NSString *output) {
    NSMutableArray *rows = [NSMutableArray array];
    for (NSString *line in [output componentsSeparatedByString:@"\n"]) {
        int count, width;
        double exact, hnsw, recall;
        int fields = sscanf(line.UTF8String, "%d\t%d\t%*f\t%lf\t%lf\t%*f\t%lf",
                            &count, &width, &exact, &hnsw, &recall);
        if (fields != 5 || count != 10000 || width <= 0) continue;
        if (!isfinite(exact) || !isfinite(hnsw) || !isfinite(recall) ||
            exact < 0 || hnsw < 0 || recall < 0 || recall > 1) return nil;
        [rows addObject:@[
            [NSString stringWithFormat:@"%d", width],
            [NSString stringWithFormat:@"%.1f%%", 100 * recall],
            [NSString stringWithFormat:@"%.3f ms", hnsw / 1000],
            [NSString stringWithFormat:@"%.3f ms", exact / 1000]
        ]];
    }
    return rows.count ? rows : nil;
}

static NSGridView *BenchmarkTable(NSArray<NSArray<NSString *> *> *rows) {
    NSMutableArray *all = [NSMutableArray arrayWithObject:
        @[@"Search width", @"Recall@10", @"HNSW time", @"Exact time"]];
    [all addObjectsFromArray:rows];
    NSMutableArray *views = [NSMutableArray array];
    for (NSArray<NSString *> *row in all) {
        NSMutableArray *cells = [NSMutableArray array];
        for (NSString *text in row) {
            NSTextField *cell = [NSTextField labelWithString:text];
            cell.font = views.count ? [NSFont monospacedDigitSystemFontOfSize:13 weight:NSFontWeightRegular]
                                    : [NSFont boldSystemFontOfSize:13];
            [cells addObject:cell];
        }
        [views addObject:cells];
    }
    NSGridView *grid = [NSGridView gridViewWithViews:views];
    grid.columnSpacing = 24;
    grid.rowSpacing = 8;
    return grid;
}

@interface SearchTools : NSObject
@end

@implementation SearchTools {
    NSWindow *_window;
    NSStackView *_content;
    BOOL _running;
}

- (void)showTitle:(NSString *)title detail:(NSString *)detail
             rows:(NSArray<NSArray<NSString *> *> *)rows {
    for (NSView *view in _content.arrangedSubviews.copy) [view removeFromSuperview];
    NSTextField *heading = [NSTextField labelWithString:title];
    heading.font = [NSFont systemFontOfSize:20 weight:NSFontWeightSemibold];
    [_content addArrangedSubview:heading];
    NSTextField *description = [NSTextField wrappingLabelWithString:detail];
    [_content addArrangedSubview:description];
    [description.widthAnchor constraintEqualToAnchor:_content.widthAnchor].active = YES;
    if (_running) {
        NSProgressIndicator *progress = [NSProgressIndicator new];
        progress.style = NSProgressIndicatorStyleBar;
        progress.indeterminate = YES;
        [progress setAccessibilityLabel:title];
        [_content addArrangedSubview:progress];
        [progress.widthAnchor constraintEqualToAnchor:_content.widthAnchor].active = YES;
        [progress startAnimation:nil];
    } else {
        if (rows) [_content addArrangedSubview:BenchmarkTable(rows)];
        NSButton *done = [NSButton buttonWithTitle:@"Done" target:_window action:@selector(performClose:)];
        done.keyEquivalent = @"\r";
        [_content addArrangedSubview:done];
    }
}

- (void)openWindow {
    if (!_window) {
        _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 490, 280)
            styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
            backing:NSBackingStoreBuffered defer:NO];
        _window.releasedWhenClosed = NO;
        _content = [NSStackView new];
        _content.orientation = NSUserInterfaceLayoutOrientationVertical;
        _content.alignment = NSLayoutAttributeLeading;
        _content.spacing = 14;
        _content.translatesAutoresizingMaskIntoConstraints = NO;
        NSView *root = _window.contentView;
        [root addSubview:_content];
        [NSLayoutConstraint activateConstraints:@[
            [_content.topAnchor constraintEqualToAnchor:root.topAnchor constant:22],
            [_content.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:24],
            [_content.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-24]
        ]];
        [_window center];
    }
    [_window makeKeyAndOrderFront:nil];
}

- (void)run:(BOOL)benchmark {
    [self openWindow];
    if (_running) return;
    _running = YES;
    _window.title = benchmark ? @"Search benchmark" : @"Engine tests";
    NSString *activity = benchmark ? @"Measuring search performance…" : @"Running engine tests…";
    NSString *detail = benchmark ? @"Building a 10,000-vector index and checking 100 queries."
                                 : @"Checking distances, graph links, search accuracy, and saved indexes.";
    [self showTitle:activity detail:detail rows:nil];
    NSString *name = benchmark ? @"hnsw_bench" : @"test_hnsw";
    NSURL *helpers = [NSBundle.mainBundle.bundleURL URLByAppendingPathComponent:@"Contents/Helpers"];
    NSURL *executable = [helpers URLByAppendingPathComponent:name];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSTask *task = [NSTask new];
        task.executableURL = executable;
        task.arguments = benchmark ? @[@"10000"] : @[];
        NSPipe *pipe = [NSPipe pipe];
        task.standardOutput = pipe;
        task.standardError = pipe;
        NSError *error = nil;
        NSString *failure = nil;
        NSArray *rows = nil;
        if (![task launchAndReturnError:&error]) {
            failure = [NSString stringWithFormat:@"Open a complete copy of the app and try again. %@", error.localizedDescription];
        } else {
            // Drain both streams before waiting so a full pipe cannot block the child.
            NSData *data = [pipe.fileHandleForReading readDataToEndOfFile];
            [task waitUntilExit];
            NSString *output = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            if (!output) output = @"";
            if (task.terminationStatus != 0) {
                NSString *last = [[output stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]
                                  componentsSeparatedByString:@"\n"].lastObject;
                failure = [NSString stringWithFormat:@"The tool stopped with status %d. Try the test again. %@",
                           task.terminationStatus, last ? last : @""];
            } else if (benchmark && !(rows = BenchmarkRows(output))) {
                failure = @"The benchmark returned an unreadable result. Rebuild the app and try again.";
            }
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            self->_running = NO;
            if (failure) {
                [self showTitle:@"Could not complete the run" detail:failure rows:nil];
            } else if (benchmark) {
                [self showTitle:@"Benchmark results"
                         detail:@"Mean time per query on 10,000 vectors. Recall@10 is the share of exact neighbors found."
                           rows:rows];
            } else {
                [self showTitle:@"All engine tests passed" detail:@"The engine completed all checks." rows:nil];
            }
        });
    });
}

- (void)runTests:(id)sender { [self run:NO]; }
- (void)runBenchmark:(id)sender { [self run:YES]; }
@end

void AddSearchToolsMenu(NSMenu *mainMenu) {
    // Menu items hold weak targets. Keep the controller for the app's lifetime.
    static SearchTools *tools;
    tools = [SearchTools new];
    NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Tools"];
    [menu addItemWithTitle:@"Run engine tests…" action:@selector(runTests:) keyEquivalent:@""].target = tools;
    [menu addItemWithTitle:@"Run benchmark…" action:@selector(runBenchmark:) keyEquivalent:@""].target = tools;
    NSMenuItem *item = [NSMenuItem new];
    item.submenu = menu;
    [mainMenu addItem:item];
}
