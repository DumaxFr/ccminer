/**
 * Lyra2 (v1) cuda implementation based on djm34 work
 * tpruvot@github 2015, Nanashi 08/2016 (from 1.8-r2)
 * DumaxFr@github 2018 - Dual Lyra2 for Phi2
 */

#include <stdio.h>

#include "cuda_helper.h"
#include "cuda_vector_uint2x4.h"

// Limited by shared mem max capacity (TPBx1532 <= 48kb)
// 48kb prefered to fit 2 times in 96kb max shared on sm52 & sm61)
#define PHI2LYRA2_TPB64_MAIN 32
#define PHI2LYRA2_TPB64_LDST 128

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 520
extern __device__ __device_builtin__ uint32_t __shfl(uint32_t, uint32_t, uint32_t);
#endif

#define Nrow 8
#define Ncol 8
#define memshift 3

#define BUF_COUNT 0

__device__ uint2 *DMatrix;

__device__ __forceinline__
void LD4S(uint2 res[3], const int row, const int col, const int thread, const int threads) {

#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*blockDim.x + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = *(DMatrix + d0 + j * threads * blockDim.x);
#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
#else
	if (row < BUF_COUNT)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = *(DMatrix + d0 + j * threads * blockDim.x);
	}
	else
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
	}
#endif
}

__device__ __forceinline__
void ST4S(const int row, const int col, const uint2 data[3], const int thread, const int threads) {

#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*blockDim.x + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		*(DMatrix + d0 + j * threads * blockDim.x) = data[j];

#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];

#else
	if (row < BUF_COUNT)
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + d0 + j * threads * blockDim.x) = data[j];
	}
	else
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];
	}
#endif
}

__device__ __forceinline__
uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c) {
	return __shfl(a, b, c);
}

__device__ __forceinline__ 
uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c) {
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__
void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c) {
	a1 = WarpShuffle(a1, b1, c);
	a2 = WarpShuffle(a2, b2, c);
	a3 = WarpShuffle(a3, b3, c);
}


__device__ __forceinline__
static void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; uint2 tmp = d; d.y = a.x ^ tmp.x; d.x = a.y ^ tmp.y;
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
}

__device__ __forceinline__
void round_lyra(uint2 s[4])
{
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

__device__ __forceinline__
static void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

__device__ __forceinline__
static void reduceDuplex(uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3];

    #pragma unroll
	for (int i = 0; i < Nrow; i++)
	{
		ST4S(0, Ncol - i - 1, state, thread, threads);

		round_lyra(state);
	}

	#pragma unroll 4
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, 0, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];
		ST4S(1, Ncol - i - 1, state1, thread, threads);
	}
}

__device__ __forceinline__
static void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	#pragma unroll 1
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, Ncol - i - 1, state1, thread, threads);

		// simultaneously receive data from preceding thread and send data to following thread
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);
	}
}

__device__ __forceinline__
static void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	for (int i = 0; i < Nrow; i++)
	{
		uint2 state1[3], state2[3];

		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

        #pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		// simultaneously receive data from preceding thread and send data to following thread
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);

		LD4S(state1, rowOut, i, thread, threads);

        #pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, i, state1, thread, threads);
	}
}

__device__ __forceinline__
static void reduceDuplexRowt_8(const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3], last[3];

	LD4S(state1, 2, 0, thread, threads);
	LD4S(last, rowInOut, 0, thread, threads);

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	// simultaneously receive data from preceding thread and send data to following thread
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	} else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 5)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{
		LD4S(state1, 2, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}


__global__
__launch_bounds__(PHI2LYRA2_TPB64_LDST, 8)
void cuda_phi2_lyra2_gpu_hash_32p1_1(const uint32_t threads, const uint2* const __restrict__ g_hash) {

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

        const uint2x4 blake2b_IV[2] = {
	        0xf3bcc908lu, 0x6a09e667lu,
	        0x84caa73blu, 0xbb67ae85lu,
	        0xfe94f82blu, 0x3c6ef372lu,
	        0x5f1d36f1lu, 0xa54ff53alu,
	        0xade682d1lu, 0x510e527flu,
	        0x2b3e6c1flu, 0x9b05688clu,
	        0xfb41bd6blu, 0x1f83d9ablu,
	        0x137e2179lu, 0x5be0cd19lu
        };

		uint2x4 state[4];

        const uint2* inHash = &g_hash[thread << 3];
		state[0].x = state[1].x = __ldg(&inHash[0]);
		state[0].y = state[1].y = __ldg(&inHash[1]);
		state[0].z = state[1].z = __ldg(&inHash[2]);
		state[0].w = state[1].w = __ldg(&inHash[3]);
        //state[0] = state[1] = __ldg4((uint2x4*)&inHash[0]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state);

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__
__launch_bounds__(PHI2LYRA2_TPB64_LDST, 8)
void cuda_phi2_lyra2_gpu_hash_32p2_1(const uint32_t threads, const uint2* const __restrict__ g_hash) {

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

        const uint2x4 blake2b_IV[2] = {
	        0xf3bcc908lu, 0x6a09e667lu,
	        0x84caa73blu, 0xbb67ae85lu,
	        0xfe94f82blu, 0x3c6ef372lu,
	        0x5f1d36f1lu, 0xa54ff53alu,
	        0xade682d1lu, 0x510e527flu,
	        0x2b3e6c1flu, 0x9b05688clu,
	        0xfb41bd6blu, 0x1f83d9ablu,
	        0x137e2179lu, 0x5be0cd19lu
        };

		uint2x4 state[4];

        const uint2* inHash = &g_hash[(thread << 3) + 4];
		state[0].x = state[1].x = __ldg(&inHash[0]);
		state[0].y = state[1].y = __ldg(&inHash[1]);
		state[0].z = state[1].z = __ldg(&inHash[2]);
		state[0].w = state[1].w = __ldg(&inHash[3]);
        //state[0] = state[1] = __ldg4((uint2x4*)&inHash[0]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state);

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__
__launch_bounds__(PHI2LYRA2_TPB64_MAIN, 1)
void cuda_phi2_lyra2_gpu_hash_32_2(const uint32_t threads) {

	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;
	if (thread < threads) {

		uint2 state[4];
		state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);

		reduceDuplex(state, thread, threads);
		reduceDuplexRowSetup(1, 0, 2, state, thread, threads);
		reduceDuplexRowSetup(2, 1, 3, state, thread, threads);
		reduceDuplexRowSetup(3, 0, 4, state, thread, threads);
		reduceDuplexRowSetup(4, 3, 5, state, thread, threads);
		reduceDuplexRowSetup(5, 2, 6, state, thread, threads);
		reduceDuplexRowSetup(6, 1, 7, state, thread, threads);

		uint32_t rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(7, rowa, 0, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(0, rowa, 3, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(3, rowa, 6, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(6, rowa, 1, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(1, rowa, 4, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(4, rowa, 7, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(7, rowa, 2, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt_8(rowa, state, thread, threads);

		DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(PHI2LYRA2_TPB64_LDST, 8)
void cuda_phi2_lyra2_gpu_hash_32p1_3(const uint32_t threads, uint2 *g_hash) {

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads) {

        uint2x4 state[4];

		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

        uint2* outHash = &g_hash[thread << 3];
		outHash[0] = state[0].x;
		outHash[1] = state[0].y;
		outHash[2] = state[0].z;
		outHash[3] = state[0].w;
        //*(uint2x4*)&outHash[0] = state[0]; // interleave better than vector store
	}
}

__global__
__launch_bounds__(PHI2LYRA2_TPB64_LDST, 8)
void cuda_phi2_lyra2_gpu_hash_32p2_3(const uint32_t threads, uint2 *g_hash) {

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads) {

        uint2x4 state[4];

		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

        uint2* outHash = &g_hash[(thread << 3) + 4];
		outHash[0] = state[0].x;
		outHash[1] = state[0].y;
		outHash[2] = state[0].z;
		outHash[3] = state[0].w;
        //*(uint2x4*)&outHash[0] = state[0];
	}
}


__host__
void cuda_phi2_lyra2_cpu_init(uint64_t *d_matrix) {
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void cuda_phi2_lyra2_cpu_hash_32x2(const uint32_t threads, uint32_t *d_hash) {
	uint32_t tpb = PHI2LYRA2_TPB64_MAIN;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + PHI2LYRA2_TPB64_LDST - 1) / PHI2LYRA2_TPB64_LDST);
	dim3 block2(PHI2LYRA2_TPB64_LDST);

	cuda_phi2_lyra2_gpu_hash_32p1_1 <<< grid2, block2 >>> (threads, (uint2*)d_hash);
	cuda_phi2_lyra2_gpu_hash_32_2 <<< grid1, block1, 192 * sizeof(uint2) * tpb >>> (threads);
	cuda_phi2_lyra2_gpu_hash_32p1_3 <<< grid2, block2 >>> (threads, (uint2*)d_hash);

	cuda_phi2_lyra2_gpu_hash_32p2_1 <<< grid2, block2 >>> (threads, (uint2*)d_hash);
	cuda_phi2_lyra2_gpu_hash_32_2 <<< grid1, block1, 192 * sizeof(uint2) * tpb >>> (threads);
	cuda_phi2_lyra2_gpu_hash_32p2_3 <<< grid2, block2 >>> (threads, (uint2*)d_hash);

}
