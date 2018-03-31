/**
* Echo512 cuda kernel for short messages
* 
* 80 & 64 bytes messages implemented with precalculations
* Code ready for message length from 8 to 96 bytes
*
* DumaxFr@github 2018
* Based on tpruvot 2018 - GPL code
* 
*/

#include "cuda_helper.h"
#include "../x11/cuda_x11_aes.cuh"

extern __device__ __device_builtin__ void __threadfence_block(void);


__device__ __forceinline__
void AES_2ROUND(const uint32_t* __restrict__ sharedMemory,
    uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3,
    uint32_t &k0) {
    uint32_t y0, y1, y2, y3;

    aes_round(sharedMemory,
        x0, x1, x2, x3,
        k0,
        y0, y1, y2, y3);

    aes_round(sharedMemory,
        y0, y1, y2, y3,
        x0, x1, x2, x3);

    k0++;
}

__device__ __forceinline__
void AES_2ROUND_MOVE(const uint32_t* __restrict__ sharedMemory,
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

#define ECHO_AES_SHIFT_IDX(i) ((((4 + (i >> 2) - (i % 4)) % 4) * 4 + (i % 4)) << 2)

__device__ __forceinline__
void first_echo_round(uint32_t* const sharedMemory, uint32_t* const paddedMsg, const uint32_t msgBitlength, uint32_t* W, uint32_t &k0) {

    const uint32_t precalc_start_80_0[8] = { 0xc2031f3a, 0x428a9633, 0xe2eaf6f3, 0xc9f3efc1, 0x56869a2b, 0x789c801f, 0x81cbd7b1, 0x4a7b67ca };
    const uint32_t precalc_start_64_0[8] = { 0xe7e9f5f5, 0x14b8a457, 0xdbfde1dd, 0x9ac2dea3, 0x65978b09, 0xa4213d7e, 0x265f4382, 0x34514d9e };
    const uint32_t precalc_start_xx_1 = 0xf5e7e9f5;
    const uint32_t precalc_start_xx_2 = 0xb3b36b23;
    const uint32_t precalc_start_xx_3 = 0xb3dbe7af;

    uint32_t idxWprime; // index for mixed BIG.SubWords and BIG.ShiftRows

    if (msgBitlength == 512) {
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            idxWprime = ECHO_AES_SHIFT_IDX(i);
            W[idxWprime] = precalc_start_64_0[i];
            W[idxWprime + 1] = precalc_start_xx_1;
            W[idxWprime + 2] = precalc_start_xx_2;
            W[idxWprime + 3] = precalc_start_xx_3;
        }
        k0 += 8;
    } else { // if (k0 == 640)
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            idxWprime = ECHO_AES_SHIFT_IDX(i);
            W[idxWprime] = precalc_start_80_0[i];
            W[idxWprime + 1] = precalc_start_xx_1;
            W[idxWprime + 2] = precalc_start_xx_2;
            W[idxWprime + 3] = precalc_start_xx_3;
        }
        k0 += 8;
    }
    //else {
    //    const int msgBitlength = k0;
    //    #pragma unroll 8
    //    for (int i = 0; i < 8; i++) {
    //        idxWprime = ECHO_AES_SHIFT_IDX(i);
    //        AES_2ROUND_MOVE(sharedMemory, msgBitlength, 0, 0, 0, k0, W[idxWprime], W[idxWprime + 1], W[idxWprime + 2], W[idxWprime + 3]);
    //        // possible workaround : store first result for "zeros" (not changed by AES) and perform "simplified" AES on first element. probably not worth ...
    //    }
    //}

    #pragma unroll 7
    for (int i = 8; i < 15; i++) {
        idxWprime = ECHO_AES_SHIFT_IDX(i);
        AES_2ROUND_MOVE(sharedMemory, paddedMsg[(i - 8 << 2)], paddedMsg[(i - 8 << 2) + 1], paddedMsg[(i - 8 << 2) + 2], paddedMsg[(i - 8 << 2) + 3],
            k0, W[idxWprime], W[idxWprime + 1], W[idxWprime + 2], W[idxWprime + 3]);
    }
    AES_2ROUND_MOVE(sharedMemory, msgBitlength, 0, 0, 0, k0, W[12], W[13], W[14], W[15]);

    // Mix Columns
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        #pragma unroll 4
        for (int idx = 0; idx < 64; idx += 16) {
            uint32_t a[4];
            a[0] = W[idx + i];
            a[1] = W[idx + i + 4];
            a[2] = W[idx + i + 8];
            a[3] = W[idx + i + 12];

            uint32_t ab = a[0] ^ a[1];
            uint32_t bc = a[1] ^ a[2];
            uint32_t cd = a[2] ^ a[3];

            uint32_t t, t2, t3;
            t = (ab & 0x80808080);
            t2 = (bc & 0x80808080);
            t3 = (cd & 0x80808080);

            uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
            uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
            uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

            W[idx + i] = bc ^ a[3] ^ abx;
            W[idx + i + 4] = a[0] ^ cd ^ bcx;
            W[idx + i + 8] = ab ^ a[3] ^ cdx;
            W[idx + i + 12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
        }
    }
}

__device__
static void echo_round(uint32_t* const sharedMemory, uint32_t *W, uint32_t &k0) {
    // Big Sub Words
    #pragma unroll 16
    for (int idx = 0; idx < 16; idx++) {
        AES_2ROUND(sharedMemory, W[(idx << 2) + 0], W[(idx << 2) + 1], W[(idx << 2) + 2], W[(idx << 2) + 3], k0);
    }

    // Shift Rows
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        uint32_t t[4];
        t[0] = W[i + 4];
        t[1] = W[i + 8];
        t[2] = W[i + 24];
        t[3] = W[i + 60];

        W[i + 4] = W[i + 20];
        W[i + 8] = W[i + 40];
        W[i + 24] = W[i + 56];
        W[i + 60] = W[i + 44];

        W[i + 20] = W[i + 36];
        W[i + 40] = t[1];
        W[i + 56] = t[2];
        W[i + 44] = W[i + 28];

        W[i + 28] = W[i + 12];
        W[i + 12] = t[3];
        W[i + 36] = W[i + 52];
        W[i + 52] = t[0];
    }

    // Mix Columns
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        #pragma unroll 4
        for (int idx = 0; idx < 64; idx += 16) {
            uint32_t a[4];
            a[0] = W[idx + i];
            a[1] = W[idx + i + 4];
            a[2] = W[idx + i + 8];
            a[3] = W[idx + i + 12];

            uint32_t ab = a[0] ^ a[1];
            uint32_t bc = a[1] ^ a[2];
            uint32_t cd = a[2] ^ a[3];

            uint32_t t, t2, t3;
            t = (ab & 0x80808080);
            t2 = (bc & 0x80808080);
            t3 = (cd & 0x80808080);

            uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
            uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
            uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

            W[idx + i] = bc ^ a[3] ^ abx;
            W[idx + i + 4] = a[0] ^ cd ^ bcx;
            W[idx + i + 8] = ab ^ a[3] ^ cdx;
            W[idx + i + 12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
        }
    }
}

__device__ __forceinline__
void last_echo_round(uint32_t* const sharedMemory, uint32_t *W, uint32_t &k0) {
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
    uint32_t y0, y1, y2, y3;
    uint32_t t0, t1, t2, t3;
    AES_2ROUND(sharedMemory, W[0], W[1], W[2], W[3], k0); // W0 not moving
    k0++;
    //AES_2ROUND(sharedMemory, W[8], W[9], W[10], W[11], k0); // W2 go to W10 (swapped)
    t0 = W[40]; t1 = W[41]; t2 = W[42]; t3 = W[43];
    aes_round(sharedMemory, W[8], W[9], W[10], W[11], k0, y0, y1, y2, y3);
    aes_round(sharedMemory, y0, y1, y2, y3, W[40], W[41], W[42], W[43]);
    k0 += 3;
    //AES_2ROUND(sharedMemory, W[20], W[21], W[22], W[23], k0); // W5 go to W1
    aes_round(sharedMemory, W[20], W[21], W[22], W[23], k0, y0, y1, y2, y3);
    aes_round(sharedMemory, y0, y1, y2, y3, W[4], W[5], W[6], W[7]);
    k0 += 2;
    //AES_2ROUND(sharedMemory, W[28], W[29], W[30], W[31], k0); // W7 go to W11
    aes_round(sharedMemory, W[28], W[29], W[30], W[31], k0, y0, y1, y2, y3);
    aes_round(sharedMemory, y0, y1, y2, y3, W[44], W[45], W[46], W[47]);
    k0++;
    AES_2ROUND(sharedMemory, W[32], W[33], W[34], W[35], k0); // W8 not moving
    k0++;
    //AES_2ROUND(sharedMemory, W[40], W[41], W[42], W[43], k0); // W10 go to W2 (swapped)
    aes_round(sharedMemory, t0, t1, t2, t3, k0, y0, y1, y2, y3);
    aes_round(sharedMemory, y0, y1, y2, y3, W[8], W[9], W[10], W[11]);
    k0 += 3;
    //AES_2ROUND(sharedMemory, W[52], W[53], W[54], W[55], k0); // W13 go to W9
    aes_round(sharedMemory, W[52], W[53], W[54], W[55], k0, y0, y1, y2, y3);
    aes_round(sharedMemory, y0, y1, y2, y3, W[36], W[37], W[38], W[39]);
    k0 += 2;
    //AES_2ROUND(sharedMemory, W[60], W[61], W[62], W[63], k0); // W15 go to W3
    aes_round(sharedMemory, W[60], W[61], W[62], W[63], k0, y0, y1, y2, y3);
    aes_round(sharedMemory, y0, y1, y2, y3, W[12], W[13], W[14], W[15]);

    // BIG.MixColumns
    // For Echo-512 Hsize, we only need to calculate W[0..15] (W0 to W3) and W[32..47] (W8 to W11)
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        #pragma unroll 2
        for (int idx = 0; idx < 64; idx += 32) {
            uint32_t a[4];
            a[0] = W[idx + i];
            a[1] = W[idx + i + 4];
            a[2] = W[idx + i + 8];
            a[3] = W[idx + i + 12];

            uint32_t ab = a[0] ^ a[1];
            uint32_t bc = a[1] ^ a[2];
            uint32_t cd = a[2] ^ a[3];

            uint32_t t, t2, t3;
            t = (ab & 0x80808080);
            t2 = (bc & 0x80808080);
            t3 = (cd & 0x80808080);

            uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
            uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
            uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

            W[idx + i] = bc ^ a[3] ^ abx;
            W[idx + i + 4] = a[0] ^ cd ^ bcx;
            W[idx + i + 8] = ab ^ a[3] ^ cdx;
            W[idx + i + 12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
        }
    }
}



__device__ __forceinline__
void cuda_echo_round_shortMsg(uint32_t* const __restrict__ sharedMemory, uint32_t* const inMsg, const uint32_t msgLengthByte, uint32_t* outHash) {

    uint32_t paddedMsg[29];
    uint32_t msgBitlength = msgLengthByte * 8;

    int padIdx;
    for (padIdx = 0; padIdx < (msgLengthByte >> 2); padIdx += 2)
        AS_UINT2(&paddedMsg[padIdx]) = AS_UINT2(&inMsg[padIdx]);
    paddedMsg[padIdx++] = 0x80; // Message end
    for (; padIdx < 27; padIdx++) {
        paddedMsg[padIdx] = 0;
    }
    paddedMsg[27] = 0x2000000; // Hsize for Echo-512
    uint32_t k0 = paddedMsg[28] = msgBitlength; // L : total message bitlength (128 bit representation)

    uint32_t W[64]; // 128bits 4x4 State matrix

    first_echo_round(sharedMemory, paddedMsg, msgBitlength, W, k0);
    for (int i = 1; i < 9; i++)
        echo_round(sharedMemory, W, k0);
    last_echo_round(sharedMemory, W, k0);

    // BIG.final
    #pragma unroll 4
    for (int i = 0; i < 16; i += 4) {
        outHash[i] = 512U ^ paddedMsg[i] ^ W[i] ^ W[i + 32];
        outHash[i + 1] = paddedMsg[i + 1] ^ W[i + 1] ^ W[i + 33];
        outHash[i + 2] = paddedMsg[i + 2] ^ W[i + 2] ^ W[i + 34];
        outHash[i + 3] = paddedMsg[i + 3] ^ W[i + 3] ^ W[i + 35];
    }
}



__device__ __forceinline__
void echo_gpu_init(uint32_t *const __restrict__ sharedMemory) {
    /* each thread startup will fill a uint32 */
    if (threadIdx.x < 128) {
        sharedMemory[threadIdx.x] = d_AES0[threadIdx.x];
        sharedMemory[threadIdx.x + 256] = d_AES1[threadIdx.x];
        sharedMemory[threadIdx.x + 512] = d_AES2[threadIdx.x];
        sharedMemory[threadIdx.x + 768] = d_AES3[threadIdx.x];

        sharedMemory[threadIdx.x + 64 * 2] = d_AES0[threadIdx.x + 64 * 2];
        sharedMemory[threadIdx.x + 64 * 2 + 256] = d_AES1[threadIdx.x + 64 * 2];
        sharedMemory[threadIdx.x + 64 * 2 + 512] = d_AES2[threadIdx.x + 64 * 2];
        sharedMemory[threadIdx.x + 64 * 2 + 768] = d_AES3[threadIdx.x + 64 * 2];
    }
}

__host__
void echo512_cuda_init(int thr_id) {
    aes_cpu_init(thr_id);
}

__constant__ static uint32_t c_message80[20];

__host__
void echo512_setBlock_80(void *endiandata) {
    cudaMemcpyToSymbol(c_message80, endiandata, sizeof(c_message80), 0, cudaMemcpyHostToDevice);
}



__global__ __launch_bounds__(128, 5) /* will force 96 registers to avoid stack frames on sm_50+*/
void echo512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash) {
    __shared__ uint32_t sharedMemory[1024];

    echo_gpu_init(sharedMemory);
    __threadfence_block();

    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads) {
        uint64_t hashPosition = thread;
        uint32_t *pHash = (uint32_t*)&g_hash[hashPosition << 3];

        cuda_echo_round_shortMsg(sharedMemory, pHash, 64, pHash);
    }
}

__host__
void echo512_cpu_hash_64(const uint32_t threads, uint32_t *d_hash) {
    const uint32_t threadsperblock = 128;

    dim3 grid((threads + threadsperblock - 1) / threadsperblock);
    dim3 block(threadsperblock);

    echo512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}


__global__ __launch_bounds__(128, 6) /* will force 80 registers to avoid stack frames on sm_50+ */
void echo512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t* g_hash) {
    __shared__ uint32_t sharedMemory[1024];

    echo_gpu_init(sharedMemory);
    __threadfence_block();

    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads) {
        uint64_t hashPosition = thread;
        uint32_t nouncedMsg[20];

        #pragma unroll
        for (int i = 0; i < 18; i += 2)
            AS_UINT2(&nouncedMsg[i]) = AS_UINT2(&c_message80[i]);
        nouncedMsg[18] = c_message80[18];
        nouncedMsg[19] = cuda_swab32(startNonce + thread);

        cuda_echo_round_shortMsg(sharedMemory, nouncedMsg, 80, (uint32_t*)&g_hash[hashPosition << 3]);
    }
}

__host__
void echo512_cuda_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash) {
    const uint32_t threadsperblock = 128;

    dim3 grid((threads + threadsperblock - 1) / threadsperblock);
    dim3 block(threadsperblock);

    echo512_gpu_hash_80<<<grid, block>>>(threads, startNonce, (uint64_t*)d_hash);
}
