/**
 * kernel_HDC.c — QuantHD (Ternary) Inference Baseline on RISC-V (RV32IMAFC)
 * Optimized with SWAR (SIMD Within A Register) Vertical Counting
 *
 * Pipeline:
 *   1. Load model weights from the compiled-in header (HDC_model.h)
 *   2. Use the provided test samples (HDC_test_X)
 *   3. [ROI] For each sample:
 *        a. Digitize to find intervals
 *        b. Bind (XOR) ID and Level hypervectors
 *        c. Bundle using fast SWAR Bit-Serial Addition
 *        d. Threshold counters and convert to Bipolar query
 *        e. Dot Product against Ternary Class Hypervectors
 *        f. Prediction = argmax(score)
 */

#ifdef DATASET_HAR
#include "HDC_model_har.h"
#elif defined(DATASET_DIGITS)
#include "HDC_model_digits.h"
#elif defined(DATASET_CANCER)
#include "HDC_model_cancer.h"
#else
#error "No dataset specified. Please define DATASET_HAR, DATASET_DIGITS, or DATASET_CANCER."
#endif
#include <gem5/m5ops.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Helper Functions                                                  */
/* ------------------------------------------------------------------ */

// Returns interval index (0 to HDC_LEVELS - 1) for a given feature value
static inline int get_interval(float val) {
    for (int i = 1; i <= HDC_LEVELS; i++) {
        if (val <= HDC_intervals[i]) {
            return i - 1;
        }
    }
    return HDC_LEVELS - 1;
}

// Bipolar dot product against ternary weights using Fast Bit-Serial Addition (SWAR)
static int predict_ternary(const uint32_t *query_packed) {
    int best_score = -2147483648; // Minimum 32-bit integer
    int best_class = 0;
    
    for (int c = 0; c < HDC_CLASSES; c++) {
        int offset = c * HDC_TERNARY_UINT32_PER_HV;
        
        uint32_t plus_counters[10] = {0};
        uint32_t minus_counters[10] = {0};
        
        for (int i = 0; i < HDC_TERNARY_UINT32_PER_HV; i++) {
            uint32_t chunk = HDC_Class_hvs[offset + i];
            uint32_t q16 = query_packed[i];
            
            // Interleave 16 contiguous query bits into even positions of 32 bits
            uint32_t q_spaced = q16;
            q_spaced = (q_spaced | (q_spaced << 8)) & 0x00FF00FF;
            q_spaced = (q_spaced | (q_spaced << 4)) & 0x0F0F0F0F;
            q_spaced = (q_spaced | (q_spaced << 2)) & 0x33333333;
            q_spaced = (q_spaced | (q_spaced << 1)) & 0x55555555;
            
            uint32_t t_mag = chunk & 0x55555555;
            uint32_t t_sign = (chunk >> 1) & 0x55555555;
            
            uint32_t match = q_spaced ^ t_sign;
            uint32_t plus = t_mag & match;
            uint32_t minus = t_mag & ~match;
            
            // Mask out padding dimensions in the last chunk
            if (i == HDC_TERNARY_UINT32_PER_HV - 1) {
                int remainder = HDC_D % 16;
                if (remainder != 0) {
                    uint32_t valid_mask = (1U << (remainder * 2)) - 1;
                    plus &= valid_mask;
                    minus &= valid_mask;
                }
            }
            
            // Fast Bit-Serial Addition for positive scores
            uint32_t carry = plus;
            for (int c_cnt = 0; c_cnt < 10; c_cnt++) {
                uint32_t next_carry = plus_counters[c_cnt] & carry;
                plus_counters[c_cnt] ^= carry;
                carry = next_carry;
                if (!carry) break;
            }
            
            // Fast Bit-Serial Addition for negative scores
            carry = minus;
            for (int c_cnt = 0; c_cnt < 10; c_cnt++) {
                uint32_t next_carry = minus_counters[c_cnt] & carry;
                minus_counters[c_cnt] ^= carry;
                carry = next_carry;
                if (!carry) break;
            }
        }
        
        int score = 0;
        // Horizontally sum the 16 parallel counters
        for (int bit = 0; bit < 16; bit++) {
            int p_count = 0;
            int m_count = 0;
            for (int c_cnt = 0; c_cnt < 10; c_cnt++) {
                if ((plus_counters[c_cnt] >> (bit * 2)) & 1) {
                    p_count |= (1 << c_cnt);
                }
                if ((minus_counters[c_cnt] >> (bit * 2)) & 1) {
                    m_count |= (1 << c_cnt);
                }
            }
            score += (p_count - m_count);
        }
        
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }
    return best_class;
}

/* ------------------------------------------------------------------ */
/*  Main Inference Logic                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("Starting HDC Ternary Inference (SWAR Optimized Baseline)...\n");
    printf("Parameters: D=%d, Features=%d, Levels=%d, Classes=%d\n", 
           HDC_D, HDC_FEATURES, HDC_LEVELS, HDC_CLASSES);

    int correct_predictions = 0;

    /* ----------------------------------------------------------------
     * DATA LAYOUT OPTIMIZATION: Transpose arrays to Column-Major
     * Placed outside ROI — one-time setup cost, not per-inference.
     * ---------------------------------------------------------------- */
    uint32_t* transposed_ID = (uint32_t*)malloc(HDC_FEATURES * HDC_UINT32_PER_HV * sizeof(uint32_t));
    for (int f = 0; f < HDC_FEATURES; f++) {
        for (int i = 0; i < HDC_UINT32_PER_HV; i++) {
            transposed_ID[i * HDC_FEATURES + f] = HDC_ID_hvs[f * HDC_UINT32_PER_HV + i];
        }
    }

    uint32_t* transposed_L = (uint32_t*)malloc(HDC_LEVELS * HDC_UINT32_PER_HV * sizeof(uint32_t));
    for (int lvl = 0; lvl < HDC_LEVELS; lvl++) {
        for (int i = 0; i < HDC_UINT32_PER_HV; i++) {
            transposed_L[i * HDC_LEVELS + lvl] = HDC_L_hvs[lvl * HDC_UINT32_PER_HV + i];
        }
    }

    /* ----------------------------------------------------------------
     * START GEM5 REGION OF INTEREST (ROI)
     * ---------------------------------------------------------------- */
    m5_reset_stats(0, 0);

    for (int s = 0; s < HDC_TEST_SAMPLES; s++) {
        const float *x = &HDC_test_X[s * HDC_FEATURES];
        
        /* 
         * SWAR Counter Array 
         * For each chunk of 32 dimensions, we need 10 registers 
         * to count up to 561 vertically. (2^10 = 1024)
         */
        uint32_t counters[HDC_UINT32_PER_HV][10];
        for (int i = 0; i < HDC_UINT32_PER_HV; i++) {
            for (int c = 0; c < 10; c++) {
                counters[i][c] = 0;
            }
        }
        
        // 1. Digitize, Bind, and Bundle (SWAR Fast Logic)
        // Pre-compute all intervals for this sample
        int intervals[HDC_FEATURES];
        for (int f = 0; f < HDC_FEATURES; f++) {
            float feature_val = x[f];
            __asm__ volatile ("" : "+f" (feature_val));
            intervals[f] = get_interval(feature_val);
        }

        for (int f = 0; f < HDC_FEATURES; f++) {
            int interval = intervals[f];
            
            for (int i = 0; i < HDC_UINT32_PER_HV; i++) {
                uint32_t bound = transposed_ID[i * HDC_FEATURES + f] ^ transposed_L[i * HDC_LEVELS + interval];
                
                // Fast Bit-Serial Addition (adds 32 dimensions concurrently)
                uint32_t carry = bound;
                for (int c = 0; c < 10; c++) {
                    uint32_t next_carry = counters[i][c] & carry;
                    counters[i][c] ^= carry;
                    carry = next_carry;
                    if (!carry) break; // Early termination if no bits overflow
                }
            }
        }
        
        // 2. Thresholding (Majority Voting) and Bipolar Conversion
        uint32_t query_packed[HDC_TERNARY_UINT32_PER_HV];
        for (int i = 0; i < HDC_TERNARY_UINT32_PER_HV; i++) {
            query_packed[i] = 0;
        }
        
        int threshold = HDC_FEATURES / 2;
        
        for (int i = 0; i < HDC_UINT32_PER_HV; i++) {
            for (int bit = 0; bit < 32; bit++) {
                int dim = i * 32 + bit;
                if (dim >= HDC_D) break;
                
                // Extract the vertical count for this specific dimension
                int count = 0;
                for (int c = 0; c < 10; c++) {
                    if ((counters[i][c] >> bit) & 1) {
                        count |= (1 << c);
                    }
                }
                
                // Convert to bipolar packed query
                if (count > threshold) {
                    int chunk_idx = dim / 16;
                    int bit_idx = dim % 16;
                    query_packed[chunk_idx] |= (1U << bit_idx);
                }
            }
        }
        
        // 3. Ternary Class Prediction
        int pred = predict_ternary(query_packed);
        
        if (pred == HDC_test_Y[s]) {
            correct_predictions++;
        }
    }

    m5_dump_stats(0, 0);
    /* ----------------------------------------------------------------
     * END GEM5 REGION OF INTEREST (ROI)
     * ---------------------------------------------------------------- */

    free(transposed_ID);
    free(transposed_L);

    printf("Inference completed.\n");
    printf("Accuracy on test batch: %d / %d\n", correct_predictions, HDC_TEST_SAMPLES);

    m5_exit(0);
    return 0;
}
