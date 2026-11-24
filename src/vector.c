#include "vector.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#endif

#define SQUARE(x) ((x) * (x))

Vector *vector_create(int dimensions){
    if (dimensions < 0 || (size_t)dimensions > (SIZE_MAX - sizeof(Vector)) / sizeof(float)) {
        fputs("Invalid vector dimension.\n", stderr);
        exit(EXIT_FAILURE);
    }
    Vector *vector = malloc(sizeof(Vector) + sizeof(float) * (size_t)dimensions);
    if (vector == NULL)
    {
        printf("Memory allocation for Vector failed.\n");
        exit(EXIT_FAILURE);
    }
    vector -> dim = dimensions;
    vector -> id = -1;
    return vector;
}

void vector_free(Vector *v){
    free(v);
}

void vector_set(Vector *v, int i, float val){
    if (i < 0 || i >= v->dim){
        printf("Given index is out of bounds.\n");
        exit(EXIT_FAILURE);
    }
    v->data[i] = val;
}

float vector_get(const Vector *v, int i){
    if (i < 0 || i >= v->dim){
        printf("Given index is out of bounds.\n");
        exit(EXIT_FAILURE);
    }
    return v->data[i];
}

void vector_fill_random(Vector *v){
    for (int i = 0; i < v->dim; i++){
        v->data[i] = (float) rand() / (float)RAND_MAX;
    }
    printf("Vector initialized with random values.\n");
}

void vector_print(const Vector *v){
    printf("Vector ID: %d\n", v->id);
    printf("Vector Dimensions: %d\n", v->dim);
    printf("Printing Vector...\n");
    printf("[");
    for (int i =0; i < v-> dim; i++){
        printf("%f, ", v->data[i]);
    }
    printf("]\n");
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float dist_l2_sq(const float *a, const float *b, int dim) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= dim - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        float32x4_t d0 = vsubq_f32(a0, b0);
        float32x4_t d1 = vsubq_f32(a1, b1);
        float32x4_t d2 = vsubq_f32(a2, b2);
        float32x4_t d3 = vsubq_f32(a3, b3);

        sum0 = vmlaq_f32(sum0, d0, d0);
        sum1 = vmlaq_f32(sum1, d1, d1);
        sum2 = vmlaq_f32(sum2, d2, d2);
        sum3 = vmlaq_f32(sum3, d3, d3);
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; i <= dim - 4; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t d0 = vsubq_f32(a0, b0);
        sum = vmlaq_f32(sum, d0, d0);
    }
    float dist = vaddvq_f32(sum);
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

float dist_dot(const float *a, const float *b, int dim) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= dim - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        sum0 = vmlaq_f32(sum0, a0, b0);
        sum1 = vmlaq_f32(sum1, a1, b1);
        sum2 = vmlaq_f32(sum2, a2, b2);
        sum3 = vmlaq_f32(sum3, a3, b3);
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; i <= dim - 4; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, a0, b0);
    }
    float dot = vaddvq_f32(sum);
    for (; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

float dist_cosine(const float *a, const float *b, int dim) {
    /* Double precision keeps the product of any two finite float inputs in range. */
    float64x2_t sum_aa = vdupq_n_f64(0.0);
    float64x2_t sum_bb = vdupq_n_f64(0.0);
    float64x2_t sum_ab = vdupq_n_f64(0.0);
    int i = 0;
    for (; i <= dim - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        float64x2_t a_lo = vcvt_f64_f32(vget_low_f32(va));
        float64x2_t a_hi = vcvt_high_f64_f32(va);
        float64x2_t b_lo = vcvt_f64_f32(vget_low_f32(vb));
        float64x2_t b_hi = vcvt_high_f64_f32(vb);

        sum_aa = vfmaq_f64(sum_aa, a_lo, a_lo);
        sum_aa = vfmaq_f64(sum_aa, a_hi, a_hi);
        sum_bb = vfmaq_f64(sum_bb, b_lo, b_lo);
        sum_bb = vfmaq_f64(sum_bb, b_hi, b_hi);
        sum_ab = vfmaq_f64(sum_ab, a_lo, b_lo);
        sum_ab = vfmaq_f64(sum_ab, a_hi, b_hi);
    }
    double aa = vaddvq_f64(sum_aa);
    double bb = vaddvq_f64(sum_bb);
    double ab = vaddvq_f64(sum_ab);
    for (; i < dim; i++) {
        double x = a[i], y = b[i];
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    if (aa == 0.0 || bb == 0.0) return 1.0f;
    double similarity = ab / sqrt(aa * bb);
    if (similarity > 1.0) similarity = 1.0;
    if (similarity < -1.0) similarity = -1.0;
    return (float)(1.0 - similarity);
}
#elif defined(__SSE__)
float dist_l2_sq(const float *a, const float *b, int dim) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i <= dim - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    float buf[4];
    _mm_storeu_ps(buf, sum);
    float dist = (buf[0] + buf[1]) + (buf[2] + buf[3]);
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

float dist_dot(const float *a, const float *b, int dim) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i <= dim - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    float buf[4];
    _mm_storeu_ps(buf, sum);
    float dot = (buf[0] + buf[1]) + (buf[2] + buf[3]);
    for (; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

#if defined(__SSE2__)
float dist_cosine(const float *a, const float *b, int dim) {
    /* Double precision keeps the product of any two finite float inputs in range. */
    __m128d sum_aa = _mm_setzero_pd();
    __m128d sum_bb = _mm_setzero_pd();
    __m128d sum_ab = _mm_setzero_pd();
    int i = 0;
    for (; i <= dim - 2; i += 2) {
        __m128d va = _mm_set_pd(a[i + 1], a[i]);
        __m128d vb = _mm_set_pd(b[i + 1], b[i]);
        sum_aa = _mm_add_pd(sum_aa, _mm_mul_pd(va, va));
        sum_bb = _mm_add_pd(sum_bb, _mm_mul_pd(vb, vb));
        sum_ab = _mm_add_pd(sum_ab, _mm_mul_pd(va, vb));
    }
    double buf_aa[2], buf_bb[2], buf_ab[2];
    _mm_storeu_pd(buf_aa, sum_aa);
    _mm_storeu_pd(buf_bb, sum_bb);
    _mm_storeu_pd(buf_ab, sum_ab);
    double aa = buf_aa[0] + buf_aa[1];
    double bb = buf_bb[0] + buf_bb[1];
    double ab = buf_ab[0] + buf_ab[1];
    for (; i < dim; i++) {
        double x = a[i], y = b[i];
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    if (aa == 0.0 || bb == 0.0) return 1.0f;
    double similarity = ab / sqrt(aa * bb);
    if (similarity > 1.0) similarity = 1.0;
    if (similarity < -1.0) similarity = -1.0;
    return (float)(1.0 - similarity);
}
#else
float dist_cosine(const float *a, const float *b, int dim) {
    double aa = 0.0, bb = 0.0, ab = 0.0;
    for (int i = 0; i < dim; i++) {
        double x = a[i], y = b[i];
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    if (aa == 0.0 || bb == 0.0) return 1.0f;
    double similarity = ab / sqrt(aa * bb);
    if (similarity > 1.0) similarity = 1.0;
    if (similarity < -1.0) similarity = -1.0;
    return (float)(1.0 - similarity);
}
#endif

#else
float dist_l2_sq(const float *a, const float *b, int dim){
    float dist = 0;
    for (int i =0; i < dim; i++){
        dist += SQUARE(a[i] - b[i]);
    }
    return dist;
}

float dist_dot(const float *a, const float *b, int dim){
    float dot = 0;
    for (int i = 0; i < dim; i++){
        dot += a[i] * b[i];
    }
    return dot;
}

float dist_cosine(const float *a, const float *b, int dim) {
    /* Double precision keeps the product of any two finite float inputs in range. */
    double aa = 0.0, bb = 0.0, ab = 0.0;
    for (int i = 0; i < dim; i++) {
        double x = a[i], y = b[i];
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    if (aa == 0.0 || bb == 0.0) return 1.0f;
    double similarity = ab / sqrt(aa * bb);
    if (similarity > 1.0) similarity = 1.0;
    if (similarity < -1.0) similarity = -1.0;
    return (float)(1.0 - similarity);
}
#endif
