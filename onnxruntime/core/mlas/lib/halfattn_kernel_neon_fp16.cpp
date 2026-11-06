/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfattn_kernel_neon_fp16.cpp

Abstract:

    Experimental (arm-fp16-exp) attention tile kernels for fp16 KV caches on ARM64 that read the cache in fp16 and
    accumulate in fp32: cache rows are widened in registers (FCVTL/FCVTL2) and multiplied with fp32 FMLA, so no fp32
    copy of the tile is written. Used by the grouped CPU GroupQueryAttention kernel.

--*/

#include "mlasi.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

#include <arm_neon.h>

namespace {

// C[r, k] = alpha * dot(A[r, :D], B[k, :D]) for MR rows of A (fp32) and NR rows of B (fp16), D % 8 == 0.
template <int MR, int NR>
MLAS_FORCEINLINE void
QKBlock(const float* A, size_t lda, const float16_t* B, size_t ldb, size_t D, float alpha, float* C, size_t ldc)
{
    float32x4_t acc[MR][NR];
    for (int r = 0; r < MR; ++r) {
        for (int k = 0; k < NR; ++k) {
            acc[r][k] = vdupq_n_f32(0.0f);
        }
    }
    for (size_t d = 0; d < D; d += 8) {
        float32x4_t bl[NR], bh[NR];
        for (int k = 0; k < NR; ++k) {
            const float16x8_t b = vld1q_f16(B + k * ldb + d);
            bl[k] = vcvt_f32_f16(vget_low_f16(b));
            bh[k] = vcvt_high_f32_f16(b);
        }
        for (int r = 0; r < MR; ++r) {
            const float32x4_t al = vld1q_f32(A + r * lda + d);
            const float32x4_t ah = vld1q_f32(A + r * lda + d + 4);
            for (int k = 0; k < NR; ++k) {
                acc[r][k] = vfmaq_f32(acc[r][k], al, bl[k]);
                acc[r][k] = vfmaq_f32(acc[r][k], ah, bh[k]);
            }
        }
    }
    for (int r = 0; r < MR; ++r) {
        if constexpr (NR == 4) {
            const float32x4_t s01 = vpaddq_f32(acc[r][0], acc[r][1]);
            const float32x4_t s23 = vpaddq_f32(acc[r][2], acc[r][3]);
            vst1q_f32(C + r * ldc, vmulq_n_f32(vpaddq_f32(s01, s23), alpha));
        } else {
            for (int k = 0; k < NR; ++k) {
                const float32x4_t s = vpaddq_f32(acc[r][k], acc[r][k]);
                C[r * ldc + k] = alpha * vgetq_lane_f32(vpaddq_f32(s, s), 0);
            }
        }
    }
}

template <int MR>
void
QKRows(size_t N, const float* A, size_t lda, const float16_t* B, size_t ldb, size_t D, float alpha, float* C, size_t ldc)
{
    size_t k = 0;
    for (; k + 4 <= N; k += 4) {
        QKBlock<MR, 4>(A, lda, B + k * ldb, ldb, D, alpha, C + k, ldc);
    }
    for (; k < N; ++k) {
        QKBlock<MR, 1>(A, lda, B + k * ldb, ldb, D, alpha, C + k, ldc);
    }
}

// C[r, 0:16] += sum_j A[r, j] * B[j, 0:16] for MR rows; A fp32, B fp16 rows.
template <int MR>
MLAS_FORCEINLINE void
PVBlock16(size_t K, const float* A, size_t lda, const float16_t* B, size_t ldb, float* C, size_t ldc)
{
    float32x4_t acc[MR][4];
    for (int r = 0; r < MR; ++r) {
        for (int i = 0; i < 4; ++i) {
            acc[r][i] = vld1q_f32(C + r * ldc + 4 * i);
        }
    }
    size_t j = 0;
    for (; j + 4 <= K; j += 4) {
        float32x4_t p[MR];
        for (int r = 0; r < MR; ++r) {
            p[r] = vld1q_f32(A + r * lda + j);
        }
#define MLAS_HALFATTN_PV_STEP(L)                                 \
    {                                                            \
        const float16x8_t v0 = vld1q_f16(B + (j + L) * ldb);     \
        const float16x8_t v1 = vld1q_f16(B + (j + L) * ldb + 8); \
        const float32x4_t w0 = vcvt_f32_f16(vget_low_f16(v0));   \
        const float32x4_t w1 = vcvt_high_f32_f16(v0);            \
        const float32x4_t w2 = vcvt_f32_f16(vget_low_f16(v1));   \
        const float32x4_t w3 = vcvt_high_f32_f16(v1);            \
        for (int r = 0; r < MR; ++r) {                           \
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], w0, p[r], L); \
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], w1, p[r], L); \
            acc[r][2] = vfmaq_laneq_f32(acc[r][2], w2, p[r], L); \
            acc[r][3] = vfmaq_laneq_f32(acc[r][3], w3, p[r], L); \
        }                                                        \
    }
        MLAS_HALFATTN_PV_STEP(0)
        MLAS_HALFATTN_PV_STEP(1)
        MLAS_HALFATTN_PV_STEP(2)
        MLAS_HALFATTN_PV_STEP(3)
#undef MLAS_HALFATTN_PV_STEP
    }
    for (; j < K; ++j) {
        const float16x8_t v0 = vld1q_f16(B + j * ldb);
        const float16x8_t v1 = vld1q_f16(B + j * ldb + 8);
        const float32x4_t w0 = vcvt_f32_f16(vget_low_f16(v0));
        const float32x4_t w1 = vcvt_high_f32_f16(v0);
        const float32x4_t w2 = vcvt_f32_f16(vget_low_f16(v1));
        const float32x4_t w3 = vcvt_high_f32_f16(v1);
        for (int r = 0; r < MR; ++r) {
            const float pj = A[r * lda + j];
            acc[r][0] = vfmaq_n_f32(acc[r][0], w0, pj);
            acc[r][1] = vfmaq_n_f32(acc[r][1], w1, pj);
            acc[r][2] = vfmaq_n_f32(acc[r][2], w2, pj);
            acc[r][3] = vfmaq_n_f32(acc[r][3], w3, pj);
        }
    }
    for (int r = 0; r < MR; ++r) {
        for (int i = 0; i < 4; ++i) {
            vst1q_f32(C + r * ldc + 4 * i, acc[r][i]);
        }
    }
}

template <int MR>
void
PVRows(size_t Ncols, size_t K, const float* A, size_t lda, const float16_t* B, size_t ldb, float* C, size_t ldc)
{
    for (size_t c = 0; c < Ncols; c += 16) {
        PVBlock16<MR>(K, A, lda, B + c, ldb, C + c, ldc);
    }
}

}  // namespace

void
MLASCALL
MlasHalfAttnQK(
    size_t M,
    size_t N,
    size_t D,
    float alpha,
    const float* A,
    size_t lda,
    const MLAS_FP16* B,
    size_t ldb,
    float* C,
    size_t ldc
)
{
    const float16_t* b = reinterpret_cast<const float16_t*>(B);
    size_t r = 0;
    for (; r + 4 <= M; r += 4) {
        QKRows<4>(N, A + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
    }
    switch (M - r) {
        case 3:
            QKRows<3>(N, A + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
            break;
        case 2:
            QKRows<2>(N, A + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
            break;
        case 1:
            QKRows<1>(N, A + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
            break;
        default:
            break;
    }
}

void
MLASCALL
MlasHalfAttnPV(
    size_t M,
    size_t Ncols,
    size_t K,
    const float* A,
    size_t lda,
    const MLAS_FP16* B,
    size_t ldb,
    float* C,
    size_t ldc
)
{
    const float16_t* b = reinterpret_cast<const float16_t*>(B);
    size_t r = 0;
    for (; r + 4 <= M; r += 4) {
        PVRows<4>(Ncols, K, A + r * lda, lda, b, ldb, C + r * ldc, ldc);
    }
    switch (M - r) {
        case 3:
            PVRows<3>(Ncols, K, A + r * lda, lda, b, ldb, C + r * ldc, ldc);
            break;
        case 2:
            PVRows<2>(Ncols, K, A + r * lda, lda, b, ldb, C + r * ldc, ldc);
            break;
        case 1:
            PVRows<1>(Ncols, K, A + r * lda, lda, b, ldb, C + r * ldc, ldc);
            break;
        default:
            break;
    }
}

#endif
