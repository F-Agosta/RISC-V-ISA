#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <gem5/m5ops.h> 

typedef struct {
    float gamma;
    float intercept;
    int32_t num_sv;
    int32_t num_features;
} SvmHeader;

// --- Function Declarations ---
float *generate_input_vectors(int num_vectors, int num_features);
float infer_svm(const float *input_vector, const float *support_vectors, const float *dual_coef, 
                int num_sv, int num_features, float gamma, float intercept);
static inline float fast_exp(float x);

int main(void) {
    // 1. Binary Model Loading
    FILE *file = fopen("training_data/svm_model_rv32imafc.bin", "rb");
    if (!file) {
        printf("Error opening the file\n");
        return 1;
    }

    SvmHeader header;
    fread(&header, sizeof(SvmHeader), 1, file);

    float *dual_coef = (float *)malloc(header.num_sv * sizeof(float));
    float *support_vectors = (float *)malloc(header.num_sv * header.num_features * sizeof(float));

    fread(dual_coef, sizeof(float), header.num_sv, file);
    fread(support_vectors, sizeof(float), header.num_sv * header.num_features, file);
    fclose(file);

    printf("Model uploaded: %d Support Vectors, %d Features.\n", header.num_sv, header.num_features);

    // Input Batch Generation (Single Contiguous Buffer)
    int num_vectors = 100;
    float *input_vectors = generate_input_vectors(num_vectors, header.num_features);

    // Array to store predictions
    float *predictions = (float *)malloc(num_vectors * sizeof(float));


    // --- START GEM5 REGION OF INTEREST (ROI) ---
    m5_reset_stats(0, 0);

    for (int i = 0; i < num_vectors; i++) {
        // Compute offset to pass only the i-th vector to the function
        int input_offset = i * header.num_features;
        
        predictions[i] = infer_svm(
            &input_vectors[input_offset], // Pointer to the beginning of the current vector
            support_vectors, 
            dual_coef, 
            header.num_sv, 
            header.num_features, 
            header.gamma, 
            header.intercept
        );
    }

    m5_dump_stats(0, 0);

    printf("Inference completed on %d vectors.\n", num_vectors);

    free(dual_coef);
    free(support_vectors);
    free(input_vectors);
    free(predictions);

    m5_exit(0);
    return 0;
}

// --- Function Implementations ---

// Allocates a SINGLE 1D block and fills it with random vectors
float *generate_input_vectors(int num_vectors, int num_features) {
    float *vectors = (float *)malloc(num_vectors * num_features * sizeof(float));
    for (int i = 0; i < num_vectors * num_features; i++) {
        vectors[i] = ((float)rand() / (float)(RAND_MAX));
    }
    return vectors;
}

// Performs inference of a single vector against all support vectors
float infer_svm(const float *input_vector, const float *support_vectors, const float *dual_coef, 
                int num_sv, int num_features, float gamma, float intercept) {
    
    float decision_value = intercept;

    for (int i = 0; i < num_sv; i++) {
        float distance_sq = 0.0f;
        int sv_offset = i * num_features;
        
        for (int j = 0; j < num_features; j++) {
            float diff = input_vector[j] - support_vectors[sv_offset + j];
            distance_sq += diff * diff;
        }

        decision_value += dual_coef[i] * fast_exp(-gamma * distance_sq);
    }

    return decision_value;
}

// Schraudolph approximation for e^x
static inline float fast_exp(float x) {
    union { float f; uint32_t i; } ecole;
    ecole.i = (uint32_t)(12102203.0f * x + 1064866805.0f);
    return ecole.f;
}
