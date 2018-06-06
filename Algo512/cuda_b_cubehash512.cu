#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "cuda_vector_uint2x4.h"


#define CUBEHASH512_TPB80 640
#define CUBEHASH512_TPB64 640
#define CUBEHASH512_TPB64F 640

#pragma region inlines

static __device__ __host__ __forceinline__ void rrounds(uint32_t *x) {

    //#pragma unroll 2
    for (int r = 0; r < 16; r++) {

		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7);x[27] = x[27] + x[11];x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7);x[29] = x[29] + x[13];x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7);x[31] = x[31] + x[15];x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" "xor x_1jklm into x_0jklm" */
		SWAP(x[0], x[8]); x[0] ^= x[16]; x[8] ^= x[24]; SWAP(x[1], x[9]); x[1] ^= x[17]; x[9] ^= x[25];
		SWAP(x[2], x[10]); x[2] ^= x[18]; x[10] ^= x[26]; SWAP(x[3], x[11]); x[3] ^= x[19]; x[11] ^= x[27];
		SWAP(x[4], x[12]); x[4] ^= x[20]; x[12] ^= x[28]; SWAP(x[5], x[13]); x[5] ^= x[21]; x[13] ^= x[29];
		SWAP(x[6], x[14]); x[6] ^= x[22]; x[14] ^= x[30]; SWAP(x[7], x[15]); x[7] ^= x[23]; x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[16], x[18]); SWAP(x[17], x[19]); SWAP(x[20], x[22]); SWAP(x[21], x[23]); SWAP(x[24], x[26]); SWAP(x[25], x[27]); SWAP(x[28], x[30]); SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10],11);x[27] = x[27] + x[11];x[11] = ROTL32(x[11],11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12],11);x[29] = x[29] + x[13];x[13] = ROTL32(x[13],11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14],11);x[31] = x[31] + x[15];x[15] = ROTL32(x[15],11);
		/* "swap x_0j0lm with x_0j1lm"  "xor x_1jklm into x_0jklm"  */
		SWAP(x[0], x[4]); x[0] ^= x[16]; x[4] ^= x[20]; SWAP(x[1], x[5]); x[1] ^= x[17]; x[5] ^= x[21];
		SWAP(x[2], x[6]); x[2] ^= x[18]; x[6] ^= x[22]; SWAP(x[3], x[7]); x[3] ^= x[19]; x[7] ^= x[23];
		SWAP(x[8], x[12]); x[8] ^= x[24]; x[12] ^= x[28]; SWAP(x[9], x[13]); x[9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; SWAP(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]); SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]); SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);

    }
}

#pragma endregion

#pragma region CubeHash512_64

__global__
__launch_bounds__(CUBEHASH512_TPB64, 2)
void cuda_base_cubehash512_gpu_hash_64(const uint32_t threads, uint32_t *g_hash) {

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads) {

        uint32_t *Hash = (uint32_t*)&g_hash[thread << 4];

        uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
			0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
			0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
			0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
        };

		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[0]);
	    rrounds(x);

		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[8]);
	    rrounds(x);

		x[0] ^= 0x80;
		rrounds(x);

	    x[31] ^= 1;
	    #pragma unroll 10
	    for (int i = 0; i < 10; i++)
            rrounds(x);

		*(uint2x4*)&Hash[0] = *(uint2x4*)&x[0];
		*(uint2x4*)&Hash[8] = *(uint2x4*)&x[8];
	}
}

__host__
void cuda_base_cubehash512_cpu_hash_64(const uint32_t threads, uint32_t *d_hash) {

	const uint32_t threadsperblock = CUBEHASH512_TPB64;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_base_cubehash512_gpu_hash_64<<<grid, block>>>(threads, d_hash);
}

#pragma endregion

#pragma region CubeHash512_64_final

__global__
__launch_bounds__(CUBEHASH512_TPB64F, 2)
void cuda_base_cubehash512_gpu_hash_64f(const uint32_t threads, const uint32_t* __restrict__ g_hash, const uint32_t startNonce, uint32_t *resNonce, const uint64_t target) {

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads) {

        const uint32_t *Hash = &g_hash[thread << 4];

        uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
			0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
			0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
			0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
        };

		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[0]);
	    rrounds(x);

		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[8]);
	    rrounds(x);

		x[0] ^= 0x80;
		rrounds(x);

	    x[31] ^= 1;
	    #pragma unroll 10
	    for (int i = 0; i < 10; i++)
            rrounds(x);

        uint64_t check = *(uint64_t*)&x[6];
		if (check <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], startNonce + thread);
            if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

__host__
void cuda_base_cubehash512_cpu_hash_64f(const uint32_t threads, const uint32_t *d_hash, const uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = CUBEHASH512_TPB64F;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_base_cubehash512_gpu_hash_64f<<<grid, block>>>(threads, d_hash, startNonce, d_resNonce, target);
}

#pragma endregion


#pragma region CubeHash512_80

__constant__ static __align__(64) uint32_t precalc_x[32];
__constant__ static __align__(64) uint32_t c_message[3];

__host__
void cuda_base_cubehash512_setBlock_80(uint32_t* endiandata) {

    uint32_t x[32] = {
		0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
		0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
		0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
		0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
		0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
		0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
		0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
		0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
    };

    AS_UINT4(&x[0]) ^= AS_UINT4(&endiandata[0]);
	AS_UINT4(&x[4]) ^= AS_UINT4(&endiandata[4]);
	rrounds(x);

	AS_UINT4(&x[0]) ^= AS_UINT4(&endiandata[8]);
	AS_UINT4(&x[4]) ^= AS_UINT4(&endiandata[12]);
	rrounds(x);

	uint32_t message[4];
    AS_UINT4(&message[0]) = AS_UINT4(&endiandata[16]);

	cudaMemcpyToSymbol(precalc_x, x, sizeof(precalc_x), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_message, message, sizeof(c_message), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(c_PaddedMessage80, endiandata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__
__launch_bounds__(CUBEHASH512_TPB80, 2)
void cuda_base_cubehash512_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t *g_outhash) {

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

		uint32_t x[32];

        *(uint2x4*)&x[0] = __ldg4((uint2x4*)&precalc_x[0]);
		*(uint2x4*)&x[8] = __ldg4((uint2x4*)&precalc_x[8]);
        *(uint2x4*)&x[16] = __ldg4((uint2x4*)&precalc_x[16]);
		*(uint2x4*)&x[24] = __ldg4((uint2x4*)&precalc_x[24]);
        
        AS_UINT2(&x[0]) ^= __ldg((uint2*)&c_message[0]);
        x[2] ^= c_message[2];
        x[3] ^= cuda_swab32(startNounce + thread);
        x[4] ^= 0x80;
   	    rrounds(x);

	    x[31] ^= 1;
	    #pragma unroll 10
	    for (int i = 0; i < 10; i++)
            rrounds(x);

        uint32_t* outHash = &g_outhash[thread << 4];
		*(uint2x4*)&outHash[0] = *(uint2x4*)&x[0];
		*(uint2x4*)&outHash[8] = *(uint2x4*)&x[8];
	}
}

__host__
void cuda_base_cubehash512_cpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash) {

	const uint32_t threadsperblock = CUBEHASH512_TPB80;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_base_cubehash512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, d_hash);
}

#pragma endregion
