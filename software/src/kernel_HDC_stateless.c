/**
 * kernel_HDC_stateless.c — QuantHD (Ternary) Inference using the Stateful HDC FU
 *
 * Uses 3 custom RISC-V instructions (custom-2 opcode, 0x5B) to fully accelerate
 * both the encoding and inference hot loops. The baseline SWAR logic is entirely
 * replaced by dedicated hardware instructions.
 *
 * Instruction encodings (R-type, opcode=0x5B):
 *   HD_BIND_ACCUM   funct3=1  — XOR(rs1, rs2) and accumulate into internal 32-element array
 *   HD_THRESHOLD    funct3=2  — Compare accumulators against rs1, emit 32-bit packed result, clear state
 *   HD_TERNARY_DOT  funct3=3  — rd = bipolar-ternary dot product(rs1[16-bit bipolar], rs2[32-bit ternary])
 *
 * Memory Optimization: Base and Level hypervector arrays are transposed to Column-Major
 * format before the ROI to maximize cache line utilization and hardware prefetch efficiency.
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
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Inline assembly macros for the custom HDC Stateful Coprocessor    */
/* ------------------------------------------------------------------ */

#define HD_BIND_ACCUM(id, lvl) \
    __asm__ volatile (".insn r 0x5B, 1, 0, x0, %0, %1" :: "r"(id), "r"(lvl))

#define HD_THRESHOLD(thresh) ({ \
    uint32_t _res; \
    __asm__ volatile (".insn r 0x5B, 2, 0, %0, %1, x0" : "=r"(_res) : "r"(thresh)); \
    _res; \
})

#define HD_TERNARY_DOT(q16, c_packed) ({ \
    int32_t _res; \
    __asm__ volatile (".insn r 0x5B, 3, 0, %0, %1, %2" : "=r"(_res) : "r"(q16), "r"(c_packed)); \
    _res; \
})

/* ------------------------------------------------------------------ */
/*  Helper: quantization interval lookup                              */
/* ------------------------------------------------------------------ */
static inline int get_interval(float val) {
    for (int i = 1; i <= HDC_LEVELS; i++) {
        if (val <= HDC_intervals[i]) return i - 1;
    }
    return HDC_LEVELS - 1;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("Starting HDC Ternary Inference (Stateless Native FU)...\n");
    printf("Parameters: D=%d, Features=%d, Levels=%d, Classes=%d\n", 
           HDC_D, HDC_FEATURES, HDC_LEVELS, HDC_CLASSES);

    int correct_predictions = 0;

    /* ----------------------------------------------------------------
     * DATA LAYOUT OPTIMIZATION: Transpose arrays to Column-Major
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
        
        // 1. Digitize, Bind, Bundle, and Threshold (1 Hardware Cycle per Instruction)
        uint32_t query_packed[HDC_TERNARY_UINT32_PER_HV];
        for (int i = 0; i < HDC_TERNARY_UINT32_PER_HV; i++) {
            query_packed[i] = 0;
        }
        
        int threshold = HDC_FEATURES / 2;
        
        int intervals[HDC_FEATURES];
        for (int f = 0; f < HDC_FEATURES; f++) {
            intervals[f] = get_interval(x[f]);
        }
        
        for (int i = 0; i < HDC_UINT32_PER_HV; i++) {
            for (int f = 0; f < HDC_FEATURES; f++) {
                uint32_t id = transposed_ID[i * HDC_FEATURES + f];
                uint32_t lvl = transposed_L[i * HDC_LEVELS + intervals[f]];
                
                HD_BIND_ACCUM(id, lvl);
            }
            
            // Extract thresholded 32-bit chunk
            uint32_t packed_res = HD_THRESHOLD(threshold);
            
            // Map the 32-bits to the 16-bit packed bipolar query format
            for (int bit = 0; bit < 32; bit++) {
                int dim = i * 32 + bit;
                if (dim >= HDC_D) break;
                
                if ((packed_res >> bit) & 1) {
                    int chunk_idx = dim / 16;
                    int bit_idx = dim % 16;
                    query_packed[chunk_idx] |= (1U << bit_idx);
                }
            }
        }


        // 3. Inference (Ternary Match) using Stateless HD_TERNARY_DOT
        int best_class = -1;
        int best_score = -2147483648; // Minimum 32-bit integer
        
        for (int c = 0; c < HDC_CLASSES; c++) {
            int score = 0;
            int offset = c * HDC_TERNARY_UINT32_PER_HV;
            
            #pragma GCC unroll 4
            for (int i = 0; i < HDC_TERNARY_UINT32_PER_HV; i++) {
                uint32_t q16 = query_packed[i];
                uint32_t c_packed = HDC_Class_hvs[offset + i];
                
                score += HD_TERNARY_DOT(q16, c_packed);
            }
            
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_class == HDC_test_Y[s]) {
            correct_predictions++;
        }
    }

    m5_dump_stats(0, 0);
    /* ----------------------------------------------------------------
     * END GEM5 REGION OF INTEREST (ROI)
     * ---------------------------------------------------------------- */

    printf("Inference Complete.\n");
    printf("Accuracy: %.2f%% (%d / %d)\n", 
           (float)correct_predictions / HDC_TEST_SAMPLES * 100.0,
           correct_predictions, HDC_TEST_SAMPLES);

    return 0;
}
