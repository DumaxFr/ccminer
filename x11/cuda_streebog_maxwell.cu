/*
 * Streebog GOST R 34.10-2012 CUDA implementation.
 *
 * https://tools.ietf.org/html/rfc6986
 * https://en.wikipedia.org/wiki/Streebog
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * @author   Tanguy Pruvot - 2015
 * @author   Alexis Provos - 2016
 */

// Further improved with shared memory partial utilization
// Tested under CUDA7.5 toolkit for cp 5.0/5.2

#include <cuda_helper.h>
#include <cuda_vectors.h>
#include <cuda_vector_uint2x4.h>

#include "streebog_arrays.cuh"

#define STREEBOG_TPB64 128

//#define FULL_UNROLL
__device__ __forceinline__
static void GOST_FS(const uint2 shared[8][256],const uint2* const __restrict__ state, uint2* return_state)
{
	return_state[0] = __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ shared[1][__byte_perm(state[6].x,0,0x44440)]
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ shared[6][__byte_perm(state[1].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)]);

	return_state[1] = __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44441)])
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ shared[6][__byte_perm(state[1].x,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)]);

	return_state[2] = __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)])
			^ shared[6][__byte_perm(state[1].x,0,0x44442)];

	return_state[3] = __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ shared[1][__byte_perm(state[6].x,0,0x44443)]
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ __ldg(&T42[__byte_perm(state[3].x,0,0x44443)])
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)])
			^ shared[6][__byte_perm(state[1].x,0,0x44443)];

	return_state[4] = __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)])
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)]);

	return_state[5] = __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)]);

	return_state[6] = __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ shared[1][__byte_perm(state[6].y,0,0x44442)]
			^ shared[2][__byte_perm(state[5].y,0,0x44442)]
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)]);

	return_state[7] = __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44443)])
			^ shared[2][__byte_perm(state[5].y,0,0x44443)]
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_FS_LDG(const uint2 shared[8][256],const uint2 *const __restrict__ state,uint2* return_state)
{
	return_state[0] = __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44440)])
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ shared[6][__byte_perm(state[1].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)]);

	return_state[1] = __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44441)])
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)])
			^ shared[6][__byte_perm(state[1].x,0,0x44441)];

	return_state[2] = __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ shared[6][__byte_perm(state[1].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)]);

	return_state[3] = __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44443)])
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ shared[4][__byte_perm(state[3].x,0,0x44443)]
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ shared[6][__byte_perm(state[1].x,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)]);

	return_state[4] = __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)]);

	return_state[5] = __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)]);

	return_state[6] = __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44442)])
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44442)])
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)]);

	return_state[7] = __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ shared[1][__byte_perm(state[6].y,0,0x44443)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44443)])
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_E12(const uint2 shared[8][256],uint2 *const __restrict__ K, uint2 *const __restrict__ state)
{
	uint2 t[8];
	for(int i=0; i<12; i++){
		GOST_FS(shared, state, t);

		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] ^= *(uint2*)&CC[i][j];

		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j] = t[ j];

		GOST_FS_LDG(shared, K, t);

		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j]^= t[ j];

		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] = t[ j];
	}
}


#if STREEBOG_TPB64 == 128
// shared[0][] & shared[7][] are always __ldg'ed
#define LUT_GPU_INIT(sharedMemory, idx) { \
	/*sharedMemory[0][(idx<<1) + 0] = __ldg(&T02[(idx<<1) + 0]);*/ \
	/*sharedMemory[0][(idx<<1) + 1] = __ldg(&T02[(idx<<1) + 1]);*/ \
	sharedMemory[1][(idx<<1) + 0] = __ldg(&T12[(idx<<1) + 0]); \
	sharedMemory[1][(idx<<1) + 1] = __ldg(&T12[(idx<<1) + 1]); \
	sharedMemory[2][(idx<<1) + 0] = __ldg(&T22[(idx<<1) + 0]); \
	sharedMemory[2][(idx<<1) + 1] = __ldg(&T22[(idx<<1) + 1]); \
	sharedMemory[3][(idx<<1) + 0] = __ldg(&T32[(idx<<1) + 0]); \
	sharedMemory[3][(idx<<1) + 1] = __ldg(&T32[(idx<<1) + 1]); \
	sharedMemory[4][(idx<<1) + 0] = __ldg(&T42[(idx<<1) + 0]); \
	sharedMemory[4][(idx<<1) + 1] = __ldg(&T42[(idx<<1) + 1]); \
	sharedMemory[5][(idx<<1) + 0] = __ldg(&T52[(idx<<1) + 0]); \
	sharedMemory[5][(idx<<1) + 1] = __ldg(&T52[(idx<<1) + 1]); \
	sharedMemory[6][(idx<<1) + 0] = __ldg(&T62[(idx<<1) + 0]); \
	sharedMemory[6][(idx<<1) + 1] = __ldg(&T62[(idx<<1) + 1]); \
	/*sharedMemory[7][(idx<<1) + 0] = __ldg(&T72[(idx<<1) + 0]);*/ \
	/*sharedMemory[7][(idx<<1) + 1] = __ldg(&T72[(idx<<1) + 1]);*/ \
}
#elif STREEBOG_TPB64 == 256
#define LUT_GPU_INIT(sharedMemory, idx) { \
	/*sharedMemory[0][idx] = __ldg(&T02[idx]);*/ \
	sharedMemory[1][idx] = __ldg(&T12[idx]); \
	sharedMemory[2][idx] = __ldg(&T22[idx]); \
	sharedMemory[3][idx] = __ldg(&T32[idx]); \
	sharedMemory[4][idx] = __ldg(&T42[idx]); \
	sharedMemory[5][idx] = __ldg(&T52[idx]); \
	sharedMemory[6][idx] = __ldg(&T62[idx]); \
	/*sharedMemory[7][idx] = __ldg(&T72[idx]);*/ \
}
#elif STREEBOG_TPB64 < 256
#define LUT_GPU_INIT(sharedMemory, idx) { \
    if (idx < 128) { \
	    /*sharedMemory[0][(idx<<1) + 0] = __ldg(&T02[(idx<<1) + 0]);*/ \
	    /*sharedMemory[0][(idx<<1) + 1] = __ldg(&T02[(idx<<1) + 1]);*/ \
	    sharedMemory[1][(idx<<1) + 0] = __ldg(&T12[(idx<<1) + 0]); \
	    sharedMemory[1][(idx<<1) + 1] = __ldg(&T12[(idx<<1) + 1]); \
	    sharedMemory[2][(idx<<1) + 0] = __ldg(&T22[(idx<<1) + 0]); \
	    sharedMemory[2][(idx<<1) + 1] = __ldg(&T22[(idx<<1) + 1]); \
	    sharedMemory[3][(idx<<1) + 0] = __ldg(&T32[(idx<<1) + 0]); \
	    sharedMemory[3][(idx<<1) + 1] = __ldg(&T32[(idx<<1) + 1]); \
	    sharedMemory[4][(idx<<1) + 0] = __ldg(&T42[(idx<<1) + 0]); \
	    sharedMemory[4][(idx<<1) + 1] = __ldg(&T42[(idx<<1) + 1]); \
	    sharedMemory[5][(idx<<1) + 0] = __ldg(&T52[(idx<<1) + 0]); \
	    sharedMemory[5][(idx<<1) + 1] = __ldg(&T52[(idx<<1) + 1]); \
	    sharedMemory[6][(idx<<1) + 0] = __ldg(&T62[(idx<<1) + 0]); \
	    sharedMemory[6][(idx<<1) + 1] = __ldg(&T62[(idx<<1) + 1]); \
	    /*sharedMemory[7][(idx<<1) + 0] = __ldg(&T72[(idx<<1) + 0]);*/ \
	    /*sharedMemory[7][(idx<<1) + 1] = __ldg(&T72[(idx<<1) + 1]);*/ \
    } \
}
#else
#define LUT_GPU_INIT(sharedMemory, idx) { \
    if (idx < 256) { \
	    /*sharedMemory[0][idx] = __ldg(&T02[idx]);*/ \
	    sharedMemory[1][idx] = __ldg(&T12[idx]); \
	    sharedMemory[2][idx] = __ldg(&T22[idx]); \
	    sharedMemory[3][idx] = __ldg(&T32[idx]); \
	    sharedMemory[4][idx] = __ldg(&T42[idx]); \
	    sharedMemory[5][idx] = __ldg(&T52[idx]); \
	    sharedMemory[6][idx] = __ldg(&T62[idx]); \
	    /*sharedMemory[7][idx] = __ldg(&T72[idx]);*/ \
    } \
}
#endif



__global__
__launch_bounds__(STREEBOG_TPB64, 3)
void streebog_gpu_hash_64_maxwell(uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2 buf[8], t[8], temp[8], K0[8], hash[8];

	__shared__ uint2 shared[8][256];

    LUT_GPU_INIT(shared, threadIdx.x)

	__threadfence_block();

	uint64_t* inout = &g_hash[thread<<3];

	*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&inout[0]);
	*(uint2x4*)&hash[4] = __ldg4((uint2x4*)&inout[4]);


	K0[0] = vectorize(0x74a5d4ce2efc83b3);

	#pragma unroll 8
	for(int i=0;i<8;i++){
		buf[ i] = K0[ 0] ^ hash[ i];
	}

	for(int i=0; i<12; i++){
		GOST_FS(shared, buf, temp);
		#pragma unroll 8
		for(uint32_t j=0;j<8;j++){
			buf[ j] = temp[ j] ^ *(uint2*)&precomputed_values[i][j];
		}
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		buf[ j]^= hash[ j];
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		K0[ j] = buf[ j];
	}

	K0[7].y ^= 0x00020000;

	GOST_FS(shared, K0, t);

	#pragma unroll 8
	for(int i=0;i<8;i++)
		K0[ i] = t[ i];

	t[7].y ^= 0x01000000;

	GOST_E12(shared, K0, t);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		buf[ j] ^= t[ j];

	buf[7].y ^= 0x01000000;

	GOST_FS(shared, buf,K0);

	buf[7].y ^= 0x00020000;

	#pragma unroll 8
	for(int j=0;j<8;j++)
		t[ j] = K0[ j];

	t[7].y ^= 0x00020000;

	GOST_E12(shared, K0, t);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		buf[ j] ^= t[ j];

	GOST_FS(shared, buf,K0); // K = F(h)

	hash[7]+= vectorize(0x0100000000000000);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		t[ j] = K0[ j] ^ hash[ j];

	GOST_E12(shared, K0, t);

	*(uint2x4*)&inout[0] = *(uint2x4*)&t[0] ^ *(uint2x4*)&hash[0] ^ *(uint2x4*)&buf[0];
	*(uint2x4*)&inout[4] = *(uint2x4*)&t[4] ^ *(uint2x4*)&hash[4] ^ *(uint2x4*)&buf[4];
}

__host__
void streebog_hash_64_maxwell(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + STREEBOG_TPB64-1) / STREEBOG_TPB64);
	dim3 block(STREEBOG_TPB64);

	streebog_gpu_hash_64_maxwell <<<grid, block>>> ((uint64_t*)d_hash);
}
