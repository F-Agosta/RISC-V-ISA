#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <gem5/m5ops.h>

#ifndef N_FEATURES
    #define N_FEATURES 64
#endif

#ifndef N_DATAPOINTS
    #define N_DATAPOINTS 256
#endif


// Define the datapoint structure
struct vector {
    float data[N_FEATURES];
    float norm;
};


void initialize_data_points(struct vector vectors[]);
void RBF_kernel(struct vector vectors[], float matrix[][N_DATAPOINTS], float gamma);
float compute_elm(struct vector *a, struct vector *b, float gamma);
static inline float fast_exp(float x);


int main(void) {


    struct vector data[N_DATAPOINTS]; // Allocate datapoints
    float K[N_DATAPOINTS][N_DATAPOINTS]; // Allocate K similarity matrix
    float gamma = 1.0;

    initialize_data_points(data);

    m5_reset_stats(0, 0);

    RBF_kernel(data, K, gamma);

    m5_dump_stats(0, 0);



    printf("Finished execution of RBF kernel\n"
           "Matrix of size %d * %d \n"
           "Vectors with %d features \n", N_DATAPOINTS, N_DATAPOINTS, N_FEATURES);

    m5_exit(0);
    return 0;

}

void initialize_data_points(struct vector vectors[]) {

    for (int i = 0; i < N_DATAPOINTS; i++) {
        vectors[i].norm = 0;
        for (int j = 0; j < N_FEATURES; j++) {
            vectors[i].data[j] = ((float)rand() / (float)(RAND_MAX));
            vectors[i].norm += vectors[i].data[j] * vectors[i].data[j];
        }
    }
}

void RBF_kernel(struct vector vectors[], float K[][N_DATAPOINTS], float gamma) {

    for (int i = 0; i < N_DATAPOINTS; i++) {
        K[i][i] = 1.0; // The diagonal is formed by only ones
        for (int j = i + 1; j < N_DATAPOINTS; j++) {
            K[i][j] = compute_elm(&vectors[i], &vectors[j], gamma);
            K[j][i] = K[i][j]; // The matrix is symmetric
        }
    }
}

float compute_elm(struct vector *a, struct vector *b, float gamma) {

    // elm = exp(-gamma * ||a - b||^2)
    float dot_product = 0;

    for (int i = 0; i < N_FEATURES; i++) {
        dot_product += a->data[i] * b->data[i];
    }


    return fast_exp(-gamma * (a->norm + b->norm - (2.0f * dot_product)));
}

static inline float fast_exp(float x) {
    union { float f; uint32_t i; } ecole; // Union avoids the need to cast
    ecole.i = (uint32_t)(12102203.0f * x + 1064866805.0f);
    return ecole.f;
}

