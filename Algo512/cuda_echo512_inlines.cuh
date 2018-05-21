#include <stdint.h>
#include <device_launch_parameters.h>

extern __device__ __forceinline__ void aes_round(const uint32_t sharedMemory[4][256],
    const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3,
    const uint32_t k0, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3);
extern __device__ __forceinline__ void aes_round(const uint32_t sharedMemory[4][256],
    const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3,
    uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3);
extern __device__ __forceinline__ void aes_round2_cripple_23(const uint32_t sharedMemory[4][256],
    const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, uint32_t &y2, uint32_t &y3);
extern __device__ __forceinline__ void AES_2ROUND(const uint32_t sharedMemory[4][256],
    uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &k0);

__device__ __forceinline__
void AES_2ROUND_MOVE(const uint32_t sharedMemory[4][256],
    uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
    uint32_t &k0,
    uint32_t &z0, uint32_t &z1, uint32_t &z2, uint32_t &z3) {
    uint32_t y0, y1, y2, y3;

    aes_round(sharedMemory,
        x0, x1, x2, x3,
        k0,
        y0, y1, y2, y3);

    aes_round(sharedMemory,
        y0, y1, y2, y3,
        z0, z1, z2, z3);

    k0++;
}

__device__ __forceinline__
void AES_2ROUND_MOVE_CRIPPLE_23(const uint32_t sharedMemory[4][256],
    uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
    uint32_t &k0,
    uint32_t &z2, uint32_t &z3) {
    uint32_t y0, y1, y2, y3;

    aes_round(sharedMemory,
        x0, x1, x2, x3,
        k0,
        y0, y1, y2, y3);

    aes_round2_cripple_23(sharedMemory,
        y0, y1, y2, y3,
        z2, z3);

    k0++;
}

#define ECHO_AES_SHIFT_IDX(i) ((((4 + (i >> 2) - (i % 4)) % 4) * 4 + (i % 4)) << 2)

__device__ __forceinline__
void first_echo_round_64(const uint32_t sharedMemory[4][256], uint32_t* const paddedMsg, uint32_t* W, uint32_t &k0) {

    const uint32_t precalc_start_64_0[8] = { 0xe7e9f5f5, 0x14b8a457, 0xdbfde1dd, 0x9ac2dea3, 0x65978b09, 0xa4213d7e, 0x265f4382, 0x34514d9e };
    const uint32_t precalc_start_xx_1 = 0xf5e7e9f5;
    const uint32_t precalc_start_xx_2 = 0xb3b36b23;
    const uint32_t precalc_start_xx_3 = 0xb3dbe7af;
    const uint32_t precalc_end_64_12[16] = {
        0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
        0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7,
        0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751,
        0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
    };

    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        uint32_t idxWprime = ECHO_AES_SHIFT_IDX(i);
        W[idxWprime] = precalc_start_64_0[i];
        W[idxWprime + 1] = precalc_start_xx_1;
        W[idxWprime + 2] = precalc_start_xx_2;
        W[idxWprime + 3] = precalc_start_xx_3;
    }
    k0 += 8;

    #pragma unroll
    for (int i = 8; i < 12; i++) {
        uint32_t idxWprime = ECHO_AES_SHIFT_IDX(i);
        AES_2ROUND_MOVE(sharedMemory, paddedMsg[(i - 8 << 2)], paddedMsg[(i - 8 << 2) + 1], paddedMsg[(i - 8 << 2) + 2], paddedMsg[(i - 8 << 2) + 3],
            k0, W[idxWprime], W[idxWprime + 1], W[idxWprime + 2], W[idxWprime + 3]);
    }
    #pragma unroll
    for (int i = 12; i < 16; i++) {
        uint32_t idxWprime = ECHO_AES_SHIFT_IDX(i);
        W[idxWprime] = precalc_end_64_12[((i-12)<<2)];
        W[idxWprime + 1] = precalc_end_64_12[((i-12)<<2) + 1];
        W[idxWprime + 2] = precalc_end_64_12[((i-12)<<2) + 2];
        W[idxWprime + 3] = precalc_end_64_12[((i-12)<<2) + 3];
    }
    k0 += 4;

    // Mix Columns
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        #pragma unroll 4
        for (int idx = 0; idx < 64; idx += 16) {
            uint32_t a = W[idx + i];
            uint32_t b = W[idx + i + 4];
            uint32_t c = W[idx + i + 8];
            uint32_t d = W[idx + i + 12];

            uint32_t ab = a ^ b;
            uint32_t bc = b ^ c;
            uint32_t cd = c ^ d;

            uint32_t t = (ab & 0x80808080);
            uint32_t t2 = (bc & 0x80808080);
            uint32_t t3 = (cd & 0x80808080);

            uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
            uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
            uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

            W[idx + i] = bc ^ d ^ abx;
            W[idx + i + 4] = a ^ cd ^ bcx;
            W[idx + i + 8] = ab ^ d ^ cdx;
            W[idx + i + 12] = ab ^ c ^ (abx ^ bcx ^ cdx);
        }
    }
}

__device__ __forceinline__
void first_echo_round_80(const uint32_t sharedMemory[4][256], uint32_t* const paddedMsg, uint32_t* W, uint32_t &k0) {

    const uint32_t precalc_start_80_0[8] = { 0xc2031f3a, 0x428a9633, 0xe2eaf6f3, 0xc9f3efc1, 0x56869a2b, 0x789c801f, 0x81cbd7b1, 0x4a7b67ca };
    const uint32_t precalc_start_xx_1 = 0xf5e7e9f5;
    const uint32_t precalc_start_xx_2 = 0xb3b36b23;
    const uint32_t precalc_start_xx_3 = 0xb3dbe7af;
    const uint32_t precalc_end_80_13[12] = {
        0x83d3d3ab, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
        0x5d99993f, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751,
        0x57706cdc, 0xe4736c70, 0xf53fa165, 0xd6be2d00
    };

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t idxWprime = ECHO_AES_SHIFT_IDX(i);
        W[idxWprime] = precalc_start_80_0[i];
        W[idxWprime + 1] = precalc_start_xx_1;
        W[idxWprime + 2] = precalc_start_xx_2;
        W[idxWprime + 3] = precalc_start_xx_3;
    }
    k0 += 8;

    #pragma unroll
    for (int i = 8; i < 13; i++) {
        uint32_t idxWprime = ECHO_AES_SHIFT_IDX(i);
        AES_2ROUND_MOVE(sharedMemory, paddedMsg[(i - 8 << 2)], paddedMsg[(i - 8 << 2) + 1], paddedMsg[(i - 8 << 2) + 2], paddedMsg[(i - 8 << 2) + 3],
            k0, W[idxWprime], W[idxWprime + 1], W[idxWprime + 2], W[idxWprime + 3]);
    }

    #pragma unroll
    for (int i = 13; i < 16; i++) {
        uint32_t idxWprime = ECHO_AES_SHIFT_IDX(i);
        W[idxWprime] = precalc_end_80_13[((i-13)<<2)];
        W[idxWprime + 1] = precalc_end_80_13[((i-13)<<2) + 1];
        W[idxWprime + 2] = precalc_end_80_13[((i-13)<<2) + 2];
        W[idxWprime + 3] = precalc_end_80_13[((i-13)<<2) + 3];
    }
    k0 += 3;


    // Mix Columns
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        #pragma unroll 4
        for (int idx = 0; idx < 64; idx += 16) {
            uint32_t a = W[idx + i];
            uint32_t b = W[idx + i + 4];
            uint32_t c = W[idx + i + 8];
            uint32_t d = W[idx + i + 12];

            uint32_t ab = a ^ b;
            uint32_t bc = b ^ c;
            uint32_t cd = c ^ d;

            uint32_t t = (ab & 0x80808080);
            uint32_t t2 = (bc & 0x80808080);
            uint32_t t3 = (cd & 0x80808080);

            uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
            uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
            uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

            W[idx + i] = bc ^ d ^ abx;
            W[idx + i + 4] = a ^ cd ^ bcx;
            W[idx + i + 8] = ab ^ d ^ cdx;
            W[idx + i + 12] = ab ^ c ^ (abx ^ bcx ^ cdx);
        }
    }
}

__device__ __forceinline__
void last_echo_round(const uint32_t sharedMemory[4][256], uint32_t *W, uint32_t &k0) {
    // BIG.SubWords
    // Are needed for next ShiftRows : W0, W2, W5, W7, W8, W10, W13, W15
    // All others will be short-circuit but k still need to be incremented
    // BIG.ShiftRows
    // W0 to W3 and W8 to W11 only are needed for next Mix columns
    // W0 and W8 don't shift
    // W1 and W9 comes from right neighbour (ie W5 and W13)
    // W'[4..7] = W[20..23]; W'[36..39] = W[52..55];
    // W2 and W10 come from 2 columns away (ie swapped)
    // W'[8..11] = W[40..43]; W'[40..43] = W[8..11];
    // W3 and W11 comes from left neighbour (ie W15 and W7)
    // W'[12..15] = W[60..63]; W'[44..47] = W[28..31];

    // W0 not moving
    AES_2ROUND(sharedMemory, W[0], W[1], W[2], W[3], k0);
    k0++;
    // W2 go to W10 (swapped)
    uint32_t t0, t1, t2, t3;
    t0 = W[40]; t1 = W[41]; t2 = W[42]; t3 = W[43];
    AES_2ROUND_MOVE(sharedMemory, W[8], W[9], W[10], W[11], k0, W[40], W[41], W[42], W[43]);
    k0 += 2;
    // W5 go to W1
    AES_2ROUND_MOVE(sharedMemory, W[20], W[21], W[22], W[23], k0, W[4], W[5], W[6], W[7]);
    k0++;
    // W7 go to W11
    AES_2ROUND_MOVE(sharedMemory, W[28], W[29], W[30], W[31], k0, W[44], W[45], W[46], W[47]);
    // W8 not moving
    AES_2ROUND(sharedMemory, W[32], W[33], W[34], W[35], k0);
    k0++;
    // W10 go to W2 (swapped)
    AES_2ROUND_MOVE(sharedMemory, t0, t1, t2, t3, k0, W[8], W[9], W[10], W[11]);
    k0 += 2;
    // W13 go to W9
    AES_2ROUND_MOVE(sharedMemory, W[52], W[53], W[54], W[55], k0, W[36], W[37], W[38], W[39]);
    k0++;
    // W15 go to W3
    AES_2ROUND_MOVE(sharedMemory, W[60], W[61], W[62], W[63], k0, W[12], W[13], W[14], W[15]);

    // BIG.MixColumns
    // For Echo-512 Hsize, we only need to calculate W[0..15] (W0 to W3) and W[32..47] (W8 to W11)
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        #pragma unroll 2
        for (int idx = 0; idx < 64; idx += 32) {
            uint32_t a = W[idx + i];
            uint32_t b = W[idx + i + 4];
            uint32_t c = W[idx + i + 8];
            uint32_t d = W[idx + i + 12];

            uint32_t ab = a ^ b;
            uint32_t bc = b ^ c;
            uint32_t cd = c ^ d;

            uint32_t t = (ab & 0x80808080);
            uint32_t t2 = (bc & 0x80808080);
            uint32_t t3 = (cd & 0x80808080);

            uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
            uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
            uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

            W[idx + i] = bc ^ d ^ abx;
            W[idx + i + 4] = a ^ cd ^ bcx;
            W[idx + i + 8] = ab ^ d ^ cdx;
            W[idx + i + 12] = ab ^ c ^ (abx ^ bcx ^ cdx);
        }
    }
}

__device__ __forceinline__
void last_echo_round_final(const uint32_t sharedMemory[4][256], uint32_t *W, uint32_t &k0) {
    // BIG.SubWords
    // Are needed for next ShiftRows : W0, W2, W5, W7, W8, W10, W13, W15
    // All others will be short-circuit but k still need to be incremented
    // BIG.ShiftRows
    // W0 to W3 and W8 to W11 only are needed for next Mix columns
    // W0 and W8 don't shift
    // W1 and W9 comes from right neighbour (ie W5 and W13)
    // W'[4..7] = W[20..23]; W'[36..39] = W[52..55];
    // W2 and W10 come from 2 columns away (ie swapped)
    // W'[8..11] = W[40..43]; W'[40..43] = W[8..11];
    // W3 and W11 comes from left neighbour (ie W15 and W7)
    // W'[12..15] = W[60..63]; W'[44..47] = W[28..31];

    // W0 not moving
    //AES_2ROUND(sharedMemory, W[0], W[1], W[2], W[3], k0);
    AES_2ROUND_MOVE_CRIPPLE_23(sharedMemory, W[0], W[1], W[2], W[3], k0, W[2], W[3]);
    k0++;
    // W2 go to W10 (swapped)
    uint32_t t0, t1, t2, t3;
    t0 = W[40]; t1 = W[41]; t2 = W[42]; t3 = W[43];
    //AES_2ROUND_MOVE(sharedMemory, W[8], W[9], W[10], W[11], k0, W[40], W[41], W[42], W[43]);
    AES_2ROUND_MOVE_CRIPPLE_23(sharedMemory, W[8], W[9], W[10], W[11], k0, W[42], W[43]);
    k0 += 2;
    // W5 go to W1
    //AES_2ROUND_MOVE(sharedMemory, W[20], W[21], W[22], W[23], k0, W[4], W[5], W[6], W[7]);
    AES_2ROUND_MOVE_CRIPPLE_23(sharedMemory, W[20], W[21], W[22], W[23], k0, W[6], W[7]);
    k0++;
    // W7 go to W11
    //AES_2ROUND_MOVE(sharedMemory, W[28], W[29], W[30], W[31], k0, W[44], W[45], W[46], W[47]);
    AES_2ROUND_MOVE_CRIPPLE_23(sharedMemory, W[28], W[29], W[30], W[31], k0, W[46], W[47]);
    // W8 not moving
    //AES_2ROUND(sharedMemory, W[32], W[33], W[34], W[35], k0);
    AES_2ROUND_MOVE_CRIPPLE_23(sharedMemory, W[32], W[33], W[34], W[35], k0, W[34], W[35]);
    k0++;
    // W10 go to W2 (swapped)
    //AES_2ROUND_MOVE(sharedMemory, t0, t1, t2, t3, k0, W[8], W[9], W[10], W[11]);
    AES_2ROUND_MOVE_CRIPPLE_23(sharedMemory, t0, t1, t2, t3, k0, W[10], W[11]);
    k0 += 2;
    // W13 go to W9
    //AES_2ROUND_MOVE(sharedMemory, W[52], W[53], W[54], W[55], k0, W[36], W[37], W[38], W[39]);
    AES_2ROUND_MOVE_CRIPPLE_23(sharedMemory, W[52], W[53], W[54], W[55], k0, W[38], W[39]);
    k0++;
    // W15 go to W3
    //AES_2ROUND_MOVE(sharedMemory, W[60], W[61], W[62], W[63], k0, W[12], W[13], W[14], W[15]);
    aes_round(sharedMemory, W[60], W[61], W[62], W[63], k0, t0, t1, t2, t3);
    aes_round2_cripple_23(sharedMemory, t0, t1, t2, t3, W[14], W[15]);

    // BIG.MixColumns
    // For Echo-512 final, we only need to calculate W[6], W[7], W[38], W[39]
    uint32_t bc = W[6] ^ W[10];
    uint32_t tmp = (bc & 0x80808080);
    uint32_t bcx = (tmp >> 7) * 27U ^ ((bc^tmp) << 1);
    W[6] = W[2] ^ W[10] ^ W[14] ^ bcx;

    bc = W[7] ^ W[11];
    tmp = (bc & 0x80808080);
    bcx = (tmp >> 7) * 27U ^ ((bc^tmp) << 1);
    W[7] = W[3] ^ W[11] ^ W[15] ^ bcx;

    bc = W[38] ^ W[42];
    tmp = (bc & 0x80808080);
    bcx = (tmp >> 7) * 27U ^ ((bc^tmp) << 1);
    W[38] = W[34] ^ W[42] ^ W[46] ^ bcx;

    bc = W[39] ^ W[43];
    tmp = (bc & 0x80808080);
    bcx = (tmp >> 7) * 27U ^ ((bc^tmp) << 1);
    W[39] = W[35] ^ W[43] ^ W[47] ^ bcx;

}