/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
*/
#include "cuda_helper.h"
#include "cuda_vectors.h"

#define INTENSIVE_GMF
#include "aes/cuda_aes.cuh"

extern __device__ __device_builtin__ void __threadfence_block(void);

#define SHAVITE512_TPB80 384
#define SHAVITE512_TPB64 384
#define SHAVITE512_TPB64F 384


__device__ __forceinline__
static void round_3_7_11(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT(sharedMemory, &r[ 0]);
	*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
	x = p[ 2] ^ *(uint4*)&r[ 0];
	KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[4];
	x.y ^= r[5];
	x.z ^= r[6];
	x.w ^= r[7];
	KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10]^= r[6];
	r[11]^= r[7];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14]^= r[10];
	r[15]^= r[11];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 1].x ^= x.x;
	p[ 1].y ^= x.y;
	p[ 1].z ^= x.z;
	p[ 1].w ^= x.w;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[ 0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT(sharedMemory,&r[28]);
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 3] ^= x;
}

__device__ __forceinline__
static void round_4_8_12(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
	x = p[ 1] ^ *(uint4*)&r[ 0];
	AES_ROUND_NOKEY(sharedMemory, &x);

	r[ 4] ^= r[29];	r[ 5] ^= r[30];
	r[ 6] ^= r[31];	r[ 7] ^= r[ 0];

	x ^= *(uint4*)&r[ 4];
	*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[ 8];
	*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
	x = p[ 3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 2] ^= x;
}


#pragma region SHAvite512_80

__constant__ static uint32_t c_message80[20];

#if SHAVITE512_TPB80 == 128
#define AES_GPU_INIT(sharedMemory) aes_gpu_init128(sharedMemory)
#elif SHAVITE512_TPB80 == 256
#define AES_GPU_INIT(sharedMemory) aes_gpu_init256(sharedMemory);
#elif SHAVITE512_TPB80 < 256
#define AES_GPU_INIT(sharedMemory) aes_gpu_init_lt_256(sharedMemory);
#else
#define AES_GPU_INIT(sharedMemory) aes_gpu_init_mt_256(sharedMemory);
#endif

__global__
__launch_bounds__(SHAVITE512_TPB80, 2)
void shavite512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t *g_hash)
{
	__shared__ uint32_t sharedMemory[4][256];

	AES_GPU_INIT(sharedMemory);
    __threadfence_block();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[ 4];
	uint4 x;
	uint32_t r[32];

	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC,	0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47,	0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};

	if (thread < threads)
	{
		uint32_t *pHash = &g_hash[thread<<4];

        #pragma unroll
        for (int msgIdx = 0; msgIdx < 20; msgIdx += 4)
            AS_UINT4(&r[msgIdx]) = AS_UINT4(&c_message80[msgIdx]);

        r[19] = cuda_swab32(startNonce + thread);
		r[20] = 0x80; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02800000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;

        for (int i = 0; i < 4; i++)
            p[i] = AS_UINT4(&state[i * 4]);

		// round 0
		x = p[ 1] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;
		x = p[ 3] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x.x ^= 0x80;
		AES_ROUND_NOKEY(sharedMemory, &x);
		x.w ^= 0x02800000;
		AES_ROUND_NOKEY(sharedMemory, &x);
		x.w ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2]^= x;

		// 1
		KEY_EXPAND_ELT(sharedMemory, &r[ 0]);
		*(uint4*)&r[ 0]^=*(uint4*)&r[28];
		r[ 0] ^= 0x280;
		r[ 3] ^= 0xFFFFFFFF;
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);

		r[ 4] ^= r[29]; r[ 5] ^= r[30];
		r[ 6] ^= r[31]; r[ 7] ^= r[ 0];

		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);

		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);
		
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		// 2
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		r[ 7] ^= (~0x280);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;
	
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		// 3
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x^=*(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24]^=*(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x280;
		r[31] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;

		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		/* round 13 */
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x280;
		r[27] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;

        AS_UINT4(&pHash[0]) = AS_UINT4(&state[0]) ^ p[2];
        AS_UINT4(&pHash[4]) = AS_UINT4(&state[4]) ^ p[3];
        AS_UINT4(&pHash[8]) = AS_UINT4(&state[8]) ^ p[0];
        AS_UINT4(&pHash[12]) = AS_UINT4(&state[12]) ^ p[1];

	}
}

__host__
void cuda_base_shavite512_cpu_hash_80(uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	dim3 grid((threads + SHAVITE512_TPB80-1)/SHAVITE512_TPB80);
	dim3 block(SHAVITE512_TPB80);

	// note: 128 threads minimum are required to init the shared memory array
	shavite512_gpu_hash_80<<<grid, block>>>(threads, startNonce, d_hash);
}

__host__
void cuda_base_shavite512_setBlock_80(void *endiandata) {
    cudaMemcpyToSymbol(c_message80, endiandata, sizeof(c_message80), 0, cudaMemcpyHostToDevice);
}

#undef AES_GPU_INIT

#pragma endregion

#pragma region SHAvite512_64

#if SHAVITE512_TPB64 == 128
#define AES_GPU_INIT(sharedMemory) aes_gpu_init128(sharedMemory)
#elif SHAVITE512_TPB64 == 256
#define AES_GPU_INIT(sharedMemory) aes_gpu_init256(sharedMemory);
#elif SHAVITE512_TPB64 < 256
#define AES_GPU_INIT(sharedMemory) aes_gpu_init_lt_256(sharedMemory);
#else
#define AES_GPU_INIT(sharedMemory) aes_gpu_init_mt_256(sharedMemory);
#endif

__global__
__launch_bounds__(SHAVITE512_TPB64, 2)
void cuda_base_shavite512_gpu_hash_64(const uint32_t threads, uint32_t *g_hash) {
	__shared__ uint32_t sharedMemory[4][256];

	AES_GPU_INIT(sharedMemory);
    __threadfence_block();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[ 4];
	uint4 x;
	uint32_t r[32];

	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC,	0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47,	0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};

	if (thread < threads)
	{
		uint32_t *pHash = &g_hash[thread<<4];

        #pragma unroll
        for (int msgIdx = 0; msgIdx < 16; msgIdx += 4)
            AS_UINT4(&r[msgIdx]) = AS_UINT4(&pHash[msgIdx]);

        for (int i = 0; i < 4; i++)
            p[i] = AS_UINT4(&state[i * 4]);

		r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
		r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;

		/* round 0 */
		x = p[ 1] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;
		x = p[ 3];
		x.x ^= 0x80;

		AES_ROUND_NOKEY(sharedMemory, &x);

		AES_ROUND_NOKEY(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2]^= x;
		// 1
		KEY_EXPAND_ELT(sharedMemory, &r[ 0]);
		*(uint4*)&r[ 0]^=*(uint4*)&r[28];
		r[ 0] ^= 0x200;
		r[ 3] ^= 0xFFFFFFFF;
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);

		r[ 4] ^= r[29]; r[ 5] ^= r[30];
		r[ 6] ^= r[31]; r[ 7] ^= r[ 0];

		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);

		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);
		
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		// 2
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		r[ 7] ^= (~0x200);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;
	
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		// 3
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x^=*(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24]^=*(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x200;
		r[31] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;

		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		/* round 13 */
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x200;
		r[27] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;

        AS_UINT4(&pHash[0]) = AS_UINT4(&state[0]) ^ p[2];
        AS_UINT4(&pHash[4]) = AS_UINT4(&state[4]) ^ p[3];
        AS_UINT4(&pHash[8]) = AS_UINT4(&state[8]) ^ p[0];
        AS_UINT4(&pHash[12]) = AS_UINT4(&state[12]) ^ p[1];

	}
}

__host__
void cuda_base_shavite512_cpu_hash_64(uint32_t threads, uint32_t *d_hash) {

	dim3 grid((threads + SHAVITE512_TPB64-1)/SHAVITE512_TPB64);
	dim3 block(SHAVITE512_TPB64);

	cuda_base_shavite512_gpu_hash_64<<<grid, block>>>(threads, d_hash);
}

#undef AES_GPU_INIT

#pragma endregion

#pragma region SHAvite512_64_final

#if SHAVITE512_TPB64F == 128
#define AES_GPU_INIT(sharedMemory) aes_gpu_init128(sharedMemory)
#elif SHAVITE512_TPB64F == 256
#define AES_GPU_INIT(sharedMemory) aes_gpu_init256(sharedMemory);
#elif SHAVITE512_TPB64F < 256
#define AES_GPU_INIT(sharedMemory) aes_gpu_init_lt_256(sharedMemory);
#else
#define AES_GPU_INIT(sharedMemory) aes_gpu_init_mt_256(sharedMemory);
#endif

__global__
__launch_bounds__(SHAVITE512_TPB64F, 2)
void cuda_base_shavite512_gpu_hash_64_final(const uint32_t threads, uint32_t *g_hash, const uint32_t startNonce, uint32_t *resNonce, const uint64_t target) {
	__shared__ uint32_t sharedMemory[4][256];

	AES_GPU_INIT(sharedMemory);
    __threadfence_block();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[ 4];
	uint4 x;
	uint32_t r[32];

	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC,	0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47,	0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};

	if (thread < threads)
	{
		uint32_t *pHash = &g_hash[thread<<4];

        #pragma unroll
        for (int msgIdx = 0; msgIdx < 16; msgIdx += 4)
            AS_UINT4(&r[msgIdx]) = AS_UINT4(&pHash[msgIdx]);

        for (int i = 0; i < 4; i++)
            p[i] = AS_UINT4(&state[i * 4]);

		r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
		r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;

		/* round 0 */
		x = p[ 1] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;
		x = p[ 3];
		x.x ^= 0x80;

		AES_ROUND_NOKEY(sharedMemory, &x);

		AES_ROUND_NOKEY(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2]^= x;
		// 1
		KEY_EXPAND_ELT(sharedMemory, &r[ 0]);
		*(uint4*)&r[ 0]^=*(uint4*)&r[28];
		r[ 0] ^= 0x200;
		r[ 3] ^= 0xFFFFFFFF;
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);

		r[ 4] ^= r[29]; r[ 5] ^= r[30];
		r[ 6] ^= r[31]; r[ 7] ^= r[ 0];

		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);

		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);
		
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		// 2
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		r[ 7] ^= (~0x200);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;
	
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		// 3
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x^=*(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24]^=*(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x200;
		r[31] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 1] ^= x;

		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,x);
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

		/* round 13 */
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &x);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		//AES_ROUND_NOKEY(sharedMemory, &x);
        aes_round2_cripple_23(sharedMemory, x.x, x.y, x.z, x.w, x.z, x.w);

        p[3] ^= AS_UINT4(&state[4]) ^ x;

        uint64_t check = *(uint64_t*)&(p[3].z);
		if (check <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], startNonce + thread);
            if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}

	}
}

__host__
void cuda_base_shavite512_cpu_hash_64f(uint32_t threads, uint32_t *d_hash, const uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target) {

	dim3 grid((threads + SHAVITE512_TPB64F-1)/SHAVITE512_TPB64F);
	dim3 block(SHAVITE512_TPB64F);

	cuda_base_shavite512_gpu_hash_64_final<<<grid, block>>>(threads, d_hash, startNonce, d_resNonce, target);
}

#undef AES_GPU_INIT

#pragma endregion
