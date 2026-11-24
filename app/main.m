#import <AppKit/AppKit.h>
#import <math.h>
#import "SearchTools.h"
#import "WordIndex.h"

static NSTextField *label(NSString *text, CGFloat size, NSFontWeight weight) {
    NSTextField *field = [NSTextField labelWithString:text];
    field.font = [NSFont systemFontOfSize:size weight:weight];
    return field;
}

static NSStackView *row(NSArray<NSView *> *views, CGFloat spacing) {
    NSStackView *stack = [NSStackView stackViewWithViews:views];
    stack.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    stack.alignment = NSLayoutAttributeCenterY;
    stack.spacing = spacing;
    return stack;
}

@interface SearchWindow : NSObject <NSApplicationDelegate, NSSearchFieldDelegate,
                                    NSTableViewDataSource, NSTableViewDelegate>
@end

@implementation SearchWindow {
    NSWindow *_window;
    NSSearchField *_field;
    NSButton *_search;
    NSPopUpButton *_mode;
    NSButton *_verify;
    NSTextField *_dataset;
    NSTextField *_status;
    NSTableView *_table;
    NSArray<NSDictionary *> *_matches;
    WordIndex *_index;
    dispatch_queue_t _worker;
    NSUInteger _request;
}

- (void)installMenus {
    NSMenu *bar = [NSMenu new];
    NSMenuItem *appItem = [NSMenuItem new];
    [bar addItem:appItem];
    NSMenu *appMenu = [NSMenu new];
    [appMenu addItemWithTitle:@"Quit Vector Search" action:@selector(terminate:) keyEquivalent:@"q"];
    appItem.submenu = appMenu;

    NSMenuItem *editItem = [NSMenuItem new];
    [bar addItem:editItem];
    NSMenu *edit = [[NSMenu alloc] initWithTitle:@"Edit"];
    [edit addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [edit addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [edit addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [edit addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
    [edit addItem:[NSMenuItem separatorItem]];
    NSMenuItem *focus = [edit addItemWithTitle:@"Focus Search" action:@selector(focusSearch:) keyEquivalent:@"l"];
    focus.target = self;
    editItem.submenu = edit;
    AddSearchToolsMenu(bar);
    NSApp.mainMenu = bar;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [self installMenus];
    _worker = dispatch_queue_create("word-search.index", DISPATCH_QUEUE_SERIAL);
    _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 660, 540)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                   NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    _window.title = @"Vector Search";
    _window.releasedWhenClosed = NO;
    [_window setContentMinSize:NSMakeSize(560, 510)];
    [self buildContent];
    [_window center];
    [_window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    [_window makeFirstResponder:_field];
    [_field selectText:nil];

    NSURL *url = [NSBundle.mainBundle URLForResource:@"words" withExtension:@"idx"];
    dispatch_async(_worker, ^{
        WordIndex *index = url ? [[WordIndex alloc] initWithURL:url] : nil;
        dispatch_async(dispatch_get_main_queue(), ^{
            self->_index = index;
            if (!index) {
                self->_dataset.stringValue = @"Word list unavailable";
                [self setStatus:@"Could not open the word list. Rebuild or reopen the app." error:YES];
                return;
            }
            NSString *count = [NSNumberFormatter localizedStringFromNumber:@(index.count)
                numberStyle:NSNumberFormatterDecimalStyle];
            self->_dataset.stringValue = [NSString stringWithFormat:@"%@ words · GloVe word vectors", count];
            self->_search.enabled = YES;
            [self search:nil];
        });
    });
}

- (void)buildContent {
    NSTextField *title = label(@"Find related words", 28, NSFontWeightBold);
    _dataset = label(@"Opening the word list…", 13, NSFontWeightRegular);
    _dataset.textColor = NSColor.secondaryLabelColor;

    _field = [NSSearchField new];
    _field.placeholderString = @"Enter a word";
    _field.stringValue = @"violin";
    _field.controlSize = NSControlSizeLarge;
    _field.sendsWholeSearchString = YES;
    _field.target = self;
    _field.action = @selector(search:);
    _field.delegate = self;
    [_field setAccessibilityLabel:@"Word to search"];
    [_field setContentHuggingPriority:NSLayoutPriorityDefaultLow - 1 forOrientation:NSLayoutConstraintOrientationHorizontal];
    _search = [NSButton buttonWithTitle:@"Search" target:self action:@selector(search:)];
    _search.controlSize = NSControlSizeLarge;
    _search.keyEquivalent = @"\r";
    _search.enabled = NO;
    NSStackView *searchRow = row(@[_field, _search], 8);

    NSTextField *tryLabel = label(@"Try", 13, NSFontWeightRegular);
    tryLabel.textColor = NSColor.secondaryLabelColor;
    NSStackView *examples = row(@[tryLabel], 6);
    for (NSString *word in @[@"violin", @"king", @"paris"]) {
        NSButton *button = [NSButton buttonWithTitle:word target:self action:@selector(example:)];
        button.bezelStyle = NSBezelStyleRoundRect;
        [examples addArrangedSubview:button];
    }

    _mode = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    [_mode addItemsWithTitles:@[@"Quick", @"Balanced", @"Thorough"]];
    [_mode selectItemAtIndex:1];
    _mode.target = self;
    _mode.action = @selector(search:);
    _mode.toolTip = @"A wider search checks more nearby words and can find closer matches.";
    [_mode setAccessibilityLabel:@"Search mode"];
    _verify = [NSButton checkboxWithTitle:@"Compare with exact search" target:self action:@selector(search:)];
    _verify.toolTip = @"Check the results against a scan of every word in the index.";
    NSStackView *options = row(@[label(@"Search mode", 13, NSFontWeightRegular), _mode, _verify], 10);

    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(0, 0, 590, 205)];
    scroll.hasVerticalScroller = YES;
    scroll.autohidesScrollers = YES;
    scroll.borderType = NSBezelBorder;
    _table = [[NSTableView alloc] initWithFrame:scroll.bounds];
    _table.dataSource = self;
    _table.delegate = self;
    _table.rowHeight = 29;
    _table.usesAlternatingRowBackgroundColors = YES;
    _table.columnAutoresizingStyle = NSTableViewFirstColumnOnlyAutoresizingStyle;
    _table.allowsColumnReordering = NO;
    NSTableColumn *word = [[NSTableColumn alloc] initWithIdentifier:@"word"];
    word.title = @"Word";
    word.width = 425;
    word.minWidth = 130;
    [_table addTableColumn:word];
    NSTableColumn *distance = [[NSTableColumn alloc] initWithIdentifier:@"distance"];
    distance.title = @"Cosine distance";
    distance.width = 130;
    distance.resizingMask = NSTableColumnNoResizing;
    distance.headerToolTip = @"Smaller values indicate a closer match.";
    [_table addTableColumn:distance];
    scroll.documentView = _table;
    [_table setAccessibilityLabel:@"Nearest words"];
    [scroll setContentHuggingPriority:NSLayoutPriorityDefaultLow - 1 forOrientation:NSLayoutConstraintOrientationVertical];

    _status = label(@"", 13, NSFontWeightRegular);
    _status.textColor = NSColor.secondaryLabelColor;
    _status.lineBreakMode = NSLineBreakByTruncatingTail;
    _status.selectable = YES;
    NSStackView *content = [NSStackView stackViewWithViews:@[title, _dataset, searchRow, examples,
        options, label(@"Nearest words", 17, NSFontWeightSemibold), scroll, _status]];
    content.orientation = NSUserInterfaceLayoutOrientationVertical;
    content.alignment = NSLayoutAttributeLeading;
    content.spacing = 8;
    [content setCustomSpacing:3 afterView:title];
    [content setCustomSpacing:20 afterView:_dataset];
    [content setCustomSpacing:16 afterView:examples];
    [content setCustomSpacing:16 afterView:options];
    content.translatesAutoresizingMaskIntoConstraints = NO;
    NSView *body = _window.contentView;
    [body addSubview:content];
    [NSLayoutConstraint activateConstraints:@[
        [content.leadingAnchor constraintEqualToAnchor:body.leadingAnchor constant:26],
        [content.trailingAnchor constraintEqualToAnchor:body.trailingAnchor constant:-26],
        [content.topAnchor constraintEqualToAnchor:body.topAnchor constant:24],
        [content.bottomAnchor constraintEqualToAnchor:body.bottomAnchor constant:-20],
        [searchRow.widthAnchor constraintEqualToAnchor:content.widthAnchor],
        [scroll.widthAnchor constraintEqualToAnchor:content.widthAnchor],
        [scroll.heightAnchor constraintGreaterThanOrEqualToConstant:180],
        [_status.widthAnchor constraintEqualToAnchor:content.widthAnchor],
        [_status.heightAnchor constraintGreaterThanOrEqualToConstant:17]
    ]];
}

- (void)setStatus:(NSString *)message error:(BOOL)error {
    _status.stringValue = message;
    _status.textColor = error ? NSColor.systemRedColor : NSColor.secondaryLabelColor;
    _status.toolTip = message;
}

- (void)controlTextDidChange:(NSNotification *)notification {
    ++_request;
    _matches = nil;
    [_table reloadData];
    if (_index) {
        [self setStatus:_field.stringValue.length ? @"Press Return to search." : @"Enter a word or choose an example." error:NO];
    }
}

- (void)example:(NSButton *)button {
    _field.stringValue = button.title;
    [self search:nil];
    [self focusSearch:nil];
}

- (void)focusSearch:(id)sender {
    [_window makeKeyAndOrderFront:nil];
    [_window makeFirstResponder:_field];
    [_field selectText:nil];
}

- (void)search:(id)sender {
    if (!_index) return;
    NSUInteger request = ++_request;
    NSString *word = [_field.stringValue stringByTrimmingCharactersInSet:
        NSCharacterSet.whitespaceAndNewlineCharacterSet];
    _matches = nil;
    [_table reloadData];
    if (!word.length) {
        [self setStatus:@"Enter a word or choose an example." error:NO];
        return;
    }
    const NSInteger widths[] = {6, 50, 150};
    NSInteger width = widths[_mode.indexOfSelectedItem];
    BOOL verify = _verify.state == NSControlStateValueOn;
    [self setStatus:@"Searching…" error:NO];
    WordIndex *index = _index;
    dispatch_async(_worker, ^{
        NSError *error = nil;
        double recall = NAN;
        NSArray *matches = [index matchesForWord:word width:width verify:verify recall:&recall error:&error];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (request != self->_request) return;
            if (!matches) {
                [self setStatus:error ? error.localizedDescription : @"The search could not run." error:YES];
                return;
            }
            self->_matches = matches;
            [self->_table reloadData];
            NSString *message;
            if (!matches.count) {
                message = @"This word has no other words to compare.";
            } else if (verify && isfinite(recall)) {
                message = [NSString stringWithFormat:@"Exact search confirmed %ld of %lu matches for “%@”.",
                    (long)lround(recall * matches.count), (unsigned long)matches.count, word];
            } else {
                message = [NSString stringWithFormat:@"Nearest matches for “%@”. A smaller distance means a closer match.", word];
            }
            [self setStatus:message error:NO];
        });
    });
}

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView {
    return (NSInteger)_matches.count;
}

- (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)column row:(NSInteger)index {
    NSTextField *cell = [tableView makeViewWithIdentifier:column.identifier owner:self];
    if (!cell) {
        cell = label(@"", 14, NSFontWeightRegular);
        cell.identifier = column.identifier;
        cell.lineBreakMode = NSLineBreakByTruncatingTail;
    }
    NSDictionary *match = _matches[index];
    cell.stringValue = [column.identifier isEqualToString:@"word"] ? match[@"word"]
        : [NSString stringWithFormat:@"%.4f", [match[@"distance"] doubleValue]];
    return cell;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}
@end

int main(void) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        __attribute__((objc_precise_lifetime)) SearchWindow *delegate = [SearchWindow new];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
