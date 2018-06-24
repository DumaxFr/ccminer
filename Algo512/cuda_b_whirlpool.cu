/**
 * Whirlpool-512 CUDA implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014-2018 djm34, tpruvot, SP, Provos Alexis, DumaxFr
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 * @author djm34 (initial draft)
 * @author tpruvot (dual old/whirlpool modes, midstate)
 * @author SP ("final" function opt and tuning)
 * @author Provos Alexis (Applied partial shared memory utilization, precomputations, merging & tuning for 970/750ti under CUDA7.5 -> +93% increased throughput of whirlpool)
 * @author DumaxFr (64 final for X16r/X16s)
 */


//#define WHIRLPOOL_TPB80 256
#define WHIRLPOOL_TPB64 256
#define WHIRLPOOL_TPB64F 256

#include <cuda_helper.h>
#include <cuda_vectors.h>
#include <cuda_vector_uint2x4.h>
#include <cuda_uint2_utils.h>

extern __device__ __device_builtin__ void __threadfence_block(void);


#include "x15/cuda_whirlpool_tables.cuh"

#pragma region constants

__device__ __align__(128) static uint2 InitVector_RC[10];

__device__ __align__(128) static uint64_t b0[256];
__device__ __align__(128) static uint64_t b7[256];

__constant__ static uint2 precomputed_round_key_64[72];

#pragma endregion


#pragma region inlines


//--------START OF WHIRLPOOL DEVICE MACROS---------------------------------------------------------------------------
__device__ __forceinline__
void static TRANSFER(uint2 *const __restrict__ dst,const uint2 *const __restrict__ src){
	dst[0] = src[ 0];
	dst[1] = src[ 1];
	dst[2] = src[ 2];
	dst[3] = src[ 3];
	dst[4] = src[ 4];
	dst[5] = src[ 5];
	dst[6] = src[ 6];
	dst[7] = src[ 7];
}

__device__ __forceinline__
static uint2 d_ROUND_ELT_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){
	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){

	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){

	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	ret ^= c0;
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){
	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	ret ^= c0;
	return ret;
}

#pragma endregion

#pragma region Whirlpool_init

__host__
void cuda_base_whirlpool_cpu_init() {

    cudaMemcpyToSymbol(InitVector_RC, plain_RC, 10*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(precomputed_round_key_64, plain_precomputed_round_key_64, 72*sizeof(uint64_t),0, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(b0, plain_T0, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
	uint64_t table7[256];
	for(int i=0; i<256; i++) {
		table7[i] = ROTR64(plain_T0[i], 8);
	}
	cudaMemcpyToSymbol(b7, table7, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);

}

#pragma endregion


#pragma region Whirlpool_64

#if WHIRLPOOL_TPB64 == 256
#define SHARED_GPU_INIT(shared) { \
		const uint2 tmp = __ldg((uint2*)&b0[threadIdx.x]); \
		shared[0][threadIdx.x] = tmp; \
		shared[1][threadIdx.x] = ROL8(tmp); \
		shared[2][threadIdx.x] = ROL16(tmp); \
		shared[3][threadIdx.x] = ROL24(tmp); \
		shared[4][threadIdx.x] = SWAPUINT2(tmp); \
		shared[5][threadIdx.x] = ROR24(tmp); \
		shared[6][threadIdx.x] = ROR16(tmp); \
	}
#else
#define SHARED_GPU_INIT(shared) { \
        if (threadIdx.x < 256) { \
		    const uint2 tmp = __ldg((uint2*)&b0[threadIdx.x]); \
		    shared[0][threadIdx.x] = tmp; \
		    shared[1][threadIdx.x] = ROL8(tmp); \
		    shared[2][threadIdx.x] = ROL16(tmp); \
		    shared[3][threadIdx.x] = ROL24(tmp); \
		    shared[4][threadIdx.x] = SWAPUINT2(tmp); \
		    shared[5][threadIdx.x] = ROR24(tmp); \
		    shared[6][threadIdx.x] = ROR16(tmp); \
	    } \
    }
#endif

__global__
__launch_bounds__(WHIRLPOOL_TPB64, 3)
void cuda_base_whirlpool_gpu_hash_64(uint32_t threads, uint32_t *g_hash)
{
	__shared__ uint2 sharedMemory[7][256];

	SHARED_GPU_INIT(sharedMemory)

    __threadfence_block();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

		uint2 hash[8], n[8], h[ 8];
		uint2 tmp[8] = {
			{0xC0EE0B30,0x672990AF},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},
			{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828}
		};

        uint2x4 *pHash = (uint2x4 *)&g_hash[thread << 4];

		*(uint2x4*)&hash[ 0] = __ldg4(&pHash[0]);
		*(uint2x4*)&hash[ 4] = __ldg4(&pHash[1]);

		#pragma unroll 8
		for(int i=0;i<8;i++)
			n[i]=hash[i];

		tmp[ 0]^= d_ROUND_ELT(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1);
		tmp[ 1]^= d_ROUND_ELT_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[ 2]^= d_ROUND_ELT(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[ 3]^= d_ROUND_ELT_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[ 4]^= d_ROUND_ELT(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[ 5]^= d_ROUND_ELT_LDG(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[ 6]^= d_ROUND_ELT(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[ 7]^= d_ROUND_ELT_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0);
		for (int i=1; i <10; i++){
			TRANSFER(n, tmp);
			tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i-1)*8+0]);
			tmp[ 1] = d_ROUND_ELT1(    sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i-1)*8+1]);
			tmp[ 2] = d_ROUND_ELT1(    sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i-1)*8+2]);
			tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i-1)*8+3]);
			tmp[ 4] = d_ROUND_ELT1(    sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i-1)*8+4]);
			tmp[ 5] = d_ROUND_ELT1(    sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i-1)*8+5]);
			tmp[ 6] = d_ROUND_ELT1(    sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i-1)*8+6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i-1)*8+7]);
		}

		TRANSFER(h, tmp);
		#pragma unroll 8
		for (int i=0; i<8; i++)
			hash[ i] = h[i] = h[i] ^ hash[i];

        n[0].x = h[0].x ^ 0x80u;
        n[0].y = h[0].y;
        #pragma unroll 6
        for (int i = 1; i < 7; i++) {
            n[i] = h[i];
        }
        n[7].x = h[7].x;
        n[7].y = h[7].y ^ 0x20000u;

//		#pragma unroll 10
		for (int i=0; i < 10; i++) {
			tmp[ 0] = InitVector_RC[i];
			tmp[ 0]^= d_ROUND_ELT(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1);
			tmp[ 1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
			tmp[ 2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
			tmp[ 3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[ 4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
			tmp[ 5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[ 6] = d_ROUND_ELT(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
			tmp[ 7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
			TRANSFER(h, tmp);
			tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
			tmp[ 1] = d_ROUND_ELT1_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
			tmp[ 2] = d_ROUND_ELT1(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
			tmp[ 3] = d_ROUND_ELT1(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
			tmp[ 4] = d_ROUND_ELT1_LDG(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
			tmp[ 5] = d_ROUND_ELT1(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
			tmp[ 6] = d_ROUND_ELT1_LDG(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
			tmp[ 7] = d_ROUND_ELT1(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
			TRANSFER(n, tmp);
		}

		hash[0] = xor3x(hash[0], n[0], vectorize(0x80));
		hash[1] = hash[1]^ n[1];
		hash[2] = hash[2]^ n[2];
		hash[3] = hash[3]^ n[3];
		hash[4] = hash[4]^ n[4];
		hash[5] = hash[5]^ n[5];
		hash[6] = hash[6]^ n[6];
		hash[7] = xor3x(hash[7], n[7], vectorize(0x2000000000000));

		pHash[0] = *(uint2x4*)&hash[0];
		pHash[1] = *(uint2x4*)&hash[4];
	}
}

#undef SHARED_GPU_INIT

__host__
void cuda_base_whirlpool_cpu_hash_64(uint32_t threads, uint32_t *d_hash) {

    const size_t TPB = WHIRLPOOL_TPB64;

	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	cuda_base_whirlpool_gpu_hash_64 <<<grid, block>>> (threads, d_hash);
}

#pragma endregion


#pragma region Whirlpool_64_final

#if WHIRLPOOL_TPB64F == 256
#define SHARED_GPU_INIT(shared) { \
		const uint2 tmp = __ldg((uint2*)&b0[threadIdx.x]); \
		shared[0][threadIdx.x] = tmp; \
		shared[1][threadIdx.x] = ROL8(tmp); \
		shared[2][threadIdx.x] = ROL16(tmp); \
		shared[3][threadIdx.x] = ROL24(tmp); \
		shared[4][threadIdx.x] = SWAPUINT2(tmp); \
		shared[5][threadIdx.x] = ROR24(tmp); \
		shared[6][threadIdx.x] = ROR16(tmp); \
	}
#else
#define SHARED_GPU_INIT(shared) { \
        if (threadIdx.x < 256) { \
		    const uint2 tmp = __ldg((uint2*)&b0[threadIdx.x]); \
		    shared[0][threadIdx.x] = tmp; \
		    shared[1][threadIdx.x] = ROL8(tmp); \
		    shared[2][threadIdx.x] = ROL16(tmp); \
		    shared[3][threadIdx.x] = ROL24(tmp); \
		    shared[4][threadIdx.x] = SWAPUINT2(tmp); \
		    shared[5][threadIdx.x] = ROR24(tmp); \
		    shared[6][threadIdx.x] = ROR16(tmp); \
	    } \
    }
#endif

__global__
__launch_bounds__(WHIRLPOOL_TPB64F, 3)
void cuda_base_whirlpool_gpu_hash_64f(uint32_t threads, const uint32_t* __restrict__ g_hash, const uint32_t startNonce, uint32_t *resNonce, const uint64_t target)
{
	__shared__ uint2 sharedMemory[7][256];

	SHARED_GPU_INIT(sharedMemory)

    __threadfence_block();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads) {

        uint2 hash[8], n[8], h[8];
        uint2 tmp[8] = {
            {0xC0EE0B30,0x672990AF},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},
            {0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828}
        };

        uint2x4 *pHash = (uint2x4 *)&g_hash[thread << 4];

        *(uint2x4*)&hash[0] = __ldg4(&pHash[0]);
        *(uint2x4*)&hash[4] = __ldg4(&pHash[1]);

        #pragma unroll 8
        for (int i = 0; i < 8; i++)
            n[i] = hash[i];

        tmp[0] ^= d_ROUND_ELT(sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1);
        tmp[1] ^= d_ROUND_ELT_LDG(sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2);
        tmp[2] ^= d_ROUND_ELT(sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3);
        tmp[3] ^= d_ROUND_ELT_LDG(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4);
        tmp[4] ^= d_ROUND_ELT(sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5);
        tmp[5] ^= d_ROUND_ELT_LDG(sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6);
        tmp[6] ^= d_ROUND_ELT(sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7);
        tmp[7] ^= d_ROUND_ELT_LDG(sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0);
        for (int i = 1; i < 10; i++) {
            TRANSFER(n, tmp);
            tmp[0] = d_ROUND_ELT1_LDG(sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i - 1) * 8 + 0]);
            tmp[1] = d_ROUND_ELT1(sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i - 1) * 8 + 1]);
            tmp[2] = d_ROUND_ELT1(sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i - 1) * 8 + 2]);
            tmp[3] = d_ROUND_ELT1_LDG(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i - 1) * 8 + 3]);
            tmp[4] = d_ROUND_ELT1(sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i - 1) * 8 + 4]);
            tmp[5] = d_ROUND_ELT1(sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i - 1) * 8 + 5]);
            tmp[6] = d_ROUND_ELT1(sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i - 1) * 8 + 6]);
            tmp[7] = d_ROUND_ELT1_LDG(sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i - 1) * 8 + 7]);
        }
        TRANSFER(h, tmp);

        #pragma unroll 8
        for (int i = 0; i < 8; i++)
            hash[i] = h[i] = h[i] ^ hash[i];

        n[0].x = h[0].x ^ 0x80u;
        n[0].y = h[0].y;
        #pragma unroll 6
        for (int i = 1; i < 7; i++) {
            n[i] = h[i];
        }
        n[7].x = h[7].x;
        n[7].y = h[7].y ^ 0x20000u;

        for (int i = 0; i < 9; i++) {
            tmp[0] = InitVector_RC[i];
            tmp[0] ^= d_ROUND_ELT(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1);
            tmp[1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
            tmp[2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
            tmp[3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
            tmp[4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
            tmp[5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
            tmp[6] = d_ROUND_ELT(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
            tmp[7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
            TRANSFER(h, tmp);
            tmp[0] = d_ROUND_ELT1(sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
            tmp[1] = d_ROUND_ELT1_LDG(sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
            tmp[2] = d_ROUND_ELT1(sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
            tmp[3] = d_ROUND_ELT1(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
            tmp[4] = d_ROUND_ELT1_LDG(sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
            tmp[5] = d_ROUND_ELT1(sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
            tmp[6] = d_ROUND_ELT1_LDG(sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
            tmp[7] = d_ROUND_ELT1(sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
            TRANSFER(n, tmp);
        }
        tmp[3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
        hash[3] ^= d_ROUND_ELT1(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);

		if (devectorize(hash[3]) <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], startNonce + thread);
            if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

#undef SHARED_GPU_INIT

__host__
void cuda_base_whirlpool_cpu_hash_64f(uint32_t threads, const uint32_t *d_hash, const uint32_t startNonce, uint32_t * d_resNonce, const uint64_t target) {

    const size_t TPB = WHIRLPOOL_TPB64F;

	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	cuda_base_whirlpool_gpu_hash_64f <<<grid, block>>> (threads, d_hash, startNonce, d_resNonce, target);
}

#pragma endregion
