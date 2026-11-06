/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfattn_kernel_neon_fhm.cpp

Abstract:

    Experimental (arm-fp16-exp) FEAT_FHM variant of the halfattn kernels: FMLAL/FMLAL2 multiply fp16 operands and
    accumulate in fp32 with no widening instructions. Needs FEAT_FHM (e.g. Snapdragon X Oryon; not Neoverse N1).
    Both operands are fp16, so the PV product takes the probabilities rounded to fp16.

--*/

#include "mlasi.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64) && defined(__ARM_FEATURE_FP16_FML)

#include <arm_neon.h>

namespace {

template <int MR, int NR>
MLAS_FORCEINLINE void
QKBlockFhm(const float16_t* A, size_t lda, const float16_t* B, size_t ldb, size_t D, float alpha, float* C, size_t ldc)
{
    float32x4_t acc[MR][NR];
    for (int r = 0; r < MR; ++r) {
        for (int k = 0; k < NR; ++k) {
            acc[r][k] = vdupq_n_f32(0.0f);
        }
    }
    for (size_t d = 0; d < D; d += 8) {
        float16x8_t b[NR];
        for (int k = 0; k < NR; ++k) {
            b[k] = vld1q_f16(B + k * ldb + d);
        }
        for (int r = 0; r < MR; ++r) {
            const float16x8_t a = vld1q_f16(A + r * lda + d);
            for (int k = 0; k < NR; ++k) {
                acc[r][k] = vfmlalq_low_f16(acc[r][k], a, b[k]);
                acc[r][k] = vfmlalq_high_f16(acc[r][k], a, b[k]);
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
QKRowsFhm(size_t N, const float16_t* A, size_t lda, const float16_t* B, size_t ldb, size_t D, float alpha, float* C,
          size_t ldc)
{
    size_t k = 0;
    for (; k + 4 <= N; k += 4) {
        QKBlockFhm<MR, 4>(A, lda, B + k * ldb, ldb, D, alpha, C + k, ldc);
    }
    for (; k < N; ++k) {
        QKBlockFhm<MR, 1>(A, lda, B + k * ldb, ldb, D, alpha, C + k, ldc);
    }
}

// C[r, 0:16] += sum_j A[r, j] * B[j, 0:16]; A and B fp16. K % 8 == 0 on the vector path.
template <int MR>
MLAS_FORCEINLINE void
PVBlock16Fhm(size_t K, const float16_t* A, size_t lda, const float16_t* B, size_t ldb, float* C, size_t ldc)
{
    float32x4_t acc[MR][4];
    for (int r = 0; r < MR; ++r) {
        for (int i = 0; i < 4; ++i) {
            acc[r][i] = vld1q_f32(C + r * ldc + 4 * i);
        }
    }
    size_t j = 0;
    for (; j + 8 <= K; j += 8) {
        float16x8_t p[MR];
        for (int r = 0; r < MR; ++r) {
            p[r] = vld1q_f16(A + r * lda + j);
        }
#define MLAS_HALFATTN_PV_STEP_FHM(L)                                       \
    {                                                                      \
        const float16x8_t v0 = vld1q_f16(B + (j + L) * ldb);               \
        const float16x8_t v1 = vld1q_f16(B + (j + L) * ldb + 8);           \
        for (int r = 0; r < MR; ++r) {                                     \
            acc[r][0] = vfmlalq_laneq_low_f16(acc[r][0], v0, p[r], L);     \
            acc[r][1] = vfmlalq_laneq_high_f16(acc[r][1], v0, p[r], L);    \
            acc[r][2] = vfmlalq_laneq_low_f16(acc[r][2], v1, p[r], L);     \
            acc[r][3] = vfmlalq_laneq_high_f16(acc[r][3], v1, p[r], L);    \
        }                                                                  \
    }
        MLAS_HALFATTN_PV_STEP_FHM(0)
        MLAS_HALFATTN_PV_STEP_FHM(1)
        MLAS_HALFATTN_PV_STEP_FHM(2)
        MLAS_HALFATTN_PV_STEP_FHM(3)
        MLAS_HALFATTN_PV_STEP_FHM(4)
        MLAS_HALFATTN_PV_STEP_FHM(5)
        MLAS_HALFATTN_PV_STEP_FHM(6)
        MLAS_HALFATTN_PV_STEP_FHM(7)
#undef MLAS_HALFATTN_PV_STEP_FHM
    }
    for (; j < K; ++j) {
        const float16x8_t v0 = vld1q_f16(B + j * ldb);
        const float16x8_t v1 = vld1q_f16(B + j * ldb + 8);
        for (int r = 0; r < MR; ++r) {
            const float16x8_t pj = vdupq_n_f16(A[r * lda + j]);
            acc[r][0] = vfmlalq_low_f16(acc[r][0], v0, pj);
            acc[r][1] = vfmlalq_high_f16(acc[r][1], v0, pj);
            acc[r][2] = vfmlalq_low_f16(acc[r][2], v1, pj);
            acc[r][3] = vfmlalq_high_f16(acc[r][3], v1, pj);
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
PVRowsFhm(size_t Ncols, size_t K, const float16_t* A, size_t lda, const float16_t* B, size_t ldb, float* C, size_t ldc)
{
    for (size_t c = 0; c < Ncols; c += 16) {
        PVBlock16Fhm<MR>(K, A, lda, B + c, ldb, C + c, ldc);
    }
}

}  // namespace

void
MLASCALL
MlasHalfAttnQKFhm(
    size_t M,
    size_t N,
    size_t D,
    float alpha,
    const MLAS_FP16* A,
    size_t lda,
    const MLAS_FP16* B,
    size_t ldb,
    float* C,
    size_t ldc
)
{
    const float16_t* a = reinterpret_cast<const float16_t*>(A);
    const float16_t* b = reinterpret_cast<const float16_t*>(B);
    size_t r = 0;
    for (; r + 4 <= M; r += 4) {
        QKRowsFhm<4>(N, a + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
    }
    switch (M - r) {
        case 3:
            QKRowsFhm<3>(N, a + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
            break;
        case 2:
            QKRowsFhm<2>(N, a + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
            break;
        case 1:
            QKRowsFhm<1>(N, a + r * lda, lda, b, ldb, D, alpha, C + r * ldc, ldc);
            break;
        default:
            break;
    }
}

void
MLASCALL
MlasHalfAttnPVFhm(
    size_t M,
    size_t Ncols,
    size_t K,
    const MLAS_FP16* A,
    size_t lda,
    const MLAS_FP16* B,
    size_t ldb,
    float* C,
    size_t ldc
)
{
    const float16_t* a = reinterpret_cast<const float16_t*>(A);
    const float16_t* b = reinterpret_cast<const float16_t*>(B);
    size_t r = 0;
    for (; r + 4 <= M; r += 4) {
        PVRowsFhm<4>(Ncols, K, a + r * lda, lda, b, ldb, C + r * ldc, ldc);
    }
    switch (M - r) {
        case 3:
            PVRowsFhm<3>(Ncols, K, a + r * lda, lda, b, ldb, C + r * ldc, ldc);
            break;
        case 2:
            PVRowsFhm<2>(Ncols, K, a + r * lda, lda, b, ldb, C + r * ldc, ldc);
            break;
        case 1:
            PVRowsFhm<1>(Ncols, K, a + r * lda, lda, b, ldb, C + r * ldc, ldc);
            break;
        default:
            break;
    }
}

#endif
