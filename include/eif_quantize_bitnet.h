/**
 * @file eif_quantize_bitnet.h
 * @brief BitNet b1.58 Ternary {-1, 0, +1} Quantization and BitLinear Engine.
 *
 * Implements Microsoft BitNet b1.58 (1.58-bit ternary weight) arithmetic:
 * - 2-bit packing: 4 ternary weights per byte
 *   00 -> 0
 *   01 -> +1
 *   10 -> -1
 *   11 -> reserved / 0
 * - Multiplication-free matrix multiplication: inner loop contains ONLY
 *   additions and subtractions, zero hardware multipliers required!
 * - C99 compliant, MISRA friendly, zero heap allocations.
 */

#ifndef EIF_QUANTIZE_BITNET_H
#define EIF_QUANTIZE_BITNET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Unpack a single ternary value from a packed byte.
 * @param byte Byte containing 4 ternary values.
 * @param pos Position within byte (0, 1, 2, or 3).
 * @return Decoded weight in {-1, 0, +1}.
 */
int8_t eif_unpack_ternary(uint8_t byte, int pos);

/**
 * @brief Pack an array of ternary values ({-1, 0, +1}) into 2-bit packed bytes.
 * @param ternary Input array of int8 values (-1, 0, 1).
 * @param packed Output buffer for packed bytes.
 * @param count Number of ternary elements.
 * @return Number of bytes written ((count + 3) / 4).
 */
size_t eif_pack_ternary(const int8_t *ternary, uint8_t *packed, size_t count);

/**
 * @brief Dense Base-3 Ternary Packing (5 ternary weights per byte, 1.60 bits/weight).
 *
 * Utilizes 3^5 = 243 <= 255 to store 5 ternary values per byte, achieving 20.0x
 * compression over FP32 (20% denser than standard 2-bit packing).
 *
 * @param ternary Input array of int8 values (-1, 0, +1).
 * @param packed Output buffer for packed bytes.
 * @param count Number of ternary elements.
 * @return Number of bytes written ((count + 4) / 5).
 */
size_t eif_pack_ternary_dense(const int8_t *ternary, uint8_t *packed, size_t count);

/**
 * @brief Multiplication-free BitLinear Matrix-Vector Multiplication for FP32 inputs.
 *
 * Computes: output[rows] = scales[row] * (W_ternary[rows, cols] @ input[cols]) + bias[row]
 *
 * The inner loop uses ONLY additions and subtractions. No floating-point or integer
 * multiplications occur across columns.
 *
 * @param weights Packed 2-bit ternary weights (rows * ((cols + 3) / 4) bytes)
 * @param scales Per-row scale factors (gamma)
 * @param input FP32 input vector of length cols
 * @param bias Optional FP32 bias vector of length rows (can be NULL)
 * @param output Output FP32 vector of length rows
 * @param rows Number of output features
 * @param cols Number of input features
 */
void eif_matmul_bitnet_f32(const uint8_t *weights,
                           const float *scales,
                           const float *input,
                           const float *bias,
                           float *output,
                           int rows,
                           int cols);

/**
 * @brief Fast Multiplication-free BitLinear Matrix-Vector Multiplication with Dense 1.60-bit weights.
 *
 * Decodes 5 weights per byte using a 256-entry L1 lookup table (1.25 KB),
 * performing pure additions and subtractions at 20.0x compression.
 *
 * @param weights Packed base-3 ternary weights (rows * ((cols + 4) / 5) bytes)
 * @param scales Per-row scale factors (gamma)
 * @param input FP32 input vector of length cols
 * @param bias Optional FP32 bias vector of length rows (can be NULL)
 * @param output Output FP32 vector of length rows
 * @param rows Number of output features
 * @param cols Number of input features
 */
void eif_matmul_bitnet_dense_f32(const uint8_t *weights,
                                 const float *scales,
                                 const float *input,
                                 const float *bias,
                                 float *output,
                                 int rows,
                                 int cols);


/**
 * @brief Pure Integer BitLinear Matrix-Vector Multiplication for INT8 inputs.
 *
 * Computes:
 *   isum = sum(W_ternary[i, j] * input[j])  (ONLY additions and subtractions)
 *   output[i] = quantize(isum * effective_scale + bias)
 *
 * @param weights Packed 2-bit ternary weights
 * @param scale_w Weight scale gamma
 * @param input INT8 input vector
 * @param scale_in Input activation scale
 * @param bias Optional INT32 bias (can be NULL)
 * @param output Output INT8 vector
 * @param scale_out Output activation scale
 * @param rows Number of output features
 * @param cols Number of input features
 */
void eif_matmul_bitnet_int8(const uint8_t *weights,
                            float scale_w,
                            const int8_t *input,
                            float scale_in,
                            const int32_t *bias,
                            int8_t *output,
                            float scale_out,
                            int rows,
                            int cols);

/**
 * @brief Fused W1 (Gate) and W3 (Up) BitLinear Matrix-Vector Multiplication.
 * Evaluates W1 and W3 in a single OpenMP parallel region.
 */
void eif_matmul_bitnet_f32_w1w3(float *out_w1, float *out_w3,
                                const uint8_t *w1, const uint8_t *w3,
                                const float *scale_w1, const float *scale_w3,
                                const float *input, int hidden_dim, int dim);

#ifdef __cplusplus
}
#endif

#endif /* EIF_QUANTIZE_BITNET_H */
