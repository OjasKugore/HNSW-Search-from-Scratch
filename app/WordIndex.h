#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/* Use one serial queue for all searches on an index. */
@interface WordIndex : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
- (nullable instancetype)initWithURL:(NSURL *)url;
@property (nonatomic, readonly) NSInteger count;

/* Each result contains a word and its cosine distance. Recall is a fraction.
   The method sets recall to NAN when verification is off or no neighbors exist. */
- (nullable NSArray<NSDictionary *> *)matchesForWord:(NSString *)word
                                               width:(NSInteger)width
                                              verify:(BOOL)verify
                                              recall:(double *_Nullable)recall
                                               error:(NSError *_Nullable *_Nullable)error;
@end

NS_ASSUME_NONNULL_END
