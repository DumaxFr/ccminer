#include <memory.h>

#include "cuda_helper.h"

#include "./cuda_b_keccak512.h"


#define KECCAK512_TPB80 256
#define KECCAK512_TPB64 256
#define KECCAK512_TPB64F 256

#pragma region constants

static const uint64_t host_keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

#pragma endregion


#pragma region Keccak512_init

__constant__ uint64_t d_keccak_round_constants[24];

//void jackpot_keccak512_cpu_init(int thr_id, uint32_t threads);
//void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen);
//void jackpot_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);
//
__host__
void cuda_base_keccak512_cpu_init() {

	// required for the 64 bytes one
	cudaMemcpyToSymbol(d_keccak_round_constants, host_keccak_round_constants,
			sizeof(host_keccak_round_constants), 0, cudaMemcpyHostToDevice);

	//jackpot_keccak512_cpu_init(thr_id, threads);
}

#pragma endregion

#pragma region inlines

__device__ __forceinline__
static void keccak_block(uint2 *s) {

	size_t i;
	uint2 t[5], u[5], v, w;

	//for (i = 0; i < 24; i++) {
	for (i = 0; i < 23; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1]  = ROL2(s[6], 44);
		s[6]  = ROL2(s[9], 20);
		s[9]  = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2]  = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4]  = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8]  = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5]  = ROL2(s[3], 28);
		s[3]  = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7]  = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= vectorize(d_keccak_round_constants[i]);
	}
}

__device__ __forceinline__
static void last_keccak_block(uint2 *s) {

	uint2 t[5], u[5], v, w;

	/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
	t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
	t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
	t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
	t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
	t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

	/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
	u[0] = t[4] ^ ROL2(t[1], 1);
	u[1] = t[0] ^ ROL2(t[2], 1);
	u[2] = t[1] ^ ROL2(t[3], 1);
	u[3] = t[2] ^ ROL2(t[4], 1);
	u[4] = t[3] ^ ROL2(t[0], 1);

	/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
	s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
	s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
	s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
	s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
	s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

	/* rho pi: b[..] = rotl(a[..], ..) */
	//v = s[1];
	s[1]  = ROL2(s[6], 44);
	s[6]  = ROL2(s[9], 20);
	s[9]  = ROL2(s[22], 61);
	//s[22] = ROL2(s[14], 39);
	//s[14] = ROL2(s[20], 18);
	//s[20] = ROL2(s[2], 62);
	s[2]  = ROL2(s[12], 43);
	//s[12] = ROL2(s[13], 25);
	//s[13] = ROL2(s[19], 8);
	//s[19] = ROL2(s[23], 56);
	//s[23] = ROL2(s[15], 41);
	//s[15] = ROL2(s[4], 27);
	s[4]  = ROL2(s[24], 14);
	//s[24] = ROL2(s[21], 2);
	//s[21] = ROL2(s[8], 55);
	s[8]  = ROL2(s[16], 45);
	//s[16] = ROL2(s[5], 36);
	s[5]  = ROL2(s[3], 28);
	s[3]  = ROL2(s[18], 21);
	//s[18] = ROL2(s[17], 15);
	//s[17] = ROL2(s[11], 10);
	//s[11] = ROL2(s[7], 6);
	s[7]  = ROL2(s[10], 3);
	//s[10] = ROL2(v, 1);

	/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
	v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
	v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; //s[9] ^= (~v) & w;

	/* iota: a[0,0] ^= round constant */
	s[0] ^= vectorize(d_keccak_round_constants[23]);
}

__device__ __forceinline__
static void last_keccak_block_final(uint2 *s) {

	uint2 t[5], u[5];

	/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
	t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
	t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
	t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
	t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
	t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

	/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
	u[0] = t[4] ^ ROL2(t[1], 1);
	//u[1] = t[0] ^ ROL2(t[2], 1);
	//u[2] = t[1] ^ ROL2(t[3], 1);
	u[3] = t[2] ^ ROL2(t[4], 1);
	u[4] = t[3] ^ ROL2(t[0], 1);

	/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
	s[0] ^= u[0];
    //s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
	//s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
	//s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
	s[3] ^= u[3];
    //s[8] ^= u[3]; s[13] ^= u[3];
    s[18] ^= u[3];
    //s[23] ^= u[3];
	s[4] ^= u[4];
    //s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4];
    s[24] ^= u[4];

	/* rho pi: b[..] = rotl(a[..], ..) */
	//v = s[1];
	//s[1]  = ROL2(s[6], 44);
	//s[6]  = ROL2(s[9], 20);
	//s[9]  = ROL2(s[22], 61);
	//s[22] = ROL2(s[14], 39);
	//s[14] = ROL2(s[20], 18);
	//s[20] = ROL2(s[2], 62);
	//s[2]  = ROL2(s[12], 43);
	//s[12] = ROL2(s[13], 25);
	//s[13] = ROL2(s[19], 8);
	//s[19] = ROL2(s[23], 56);
	//s[23] = ROL2(s[15], 41);
	//s[15] = ROL2(s[4], 27);
	s[4]  = ROL2(s[24], 14);
	//s[24] = ROL2(s[21], 2);
	//s[21] = ROL2(s[8], 55);
	//s[8]  = ROL2(s[16], 45);
	//s[16] = ROL2(s[5], 36);
	//s[5]  = ROL2(s[3], 28);
	s[3]  = ROL2(s[18], 21);
	//s[18] = ROL2(s[17], 15);
	//s[17] = ROL2(s[11], 10);
	//s[11] = ROL2(s[7], 6);
	//s[7]  = ROL2(s[10], 3);
	//s[10] = ROL2(v, 1);

	/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
    s[3] ^= (~s[4]) & s[0];

	/* iota: a[0,0] ^= round constant */
	//s[0] ^= vectorize(d_keccak_round_constants[23]);
}

#pragma endregion


#pragma region Keccak512_80

__host__
static void keccak_block_host(uint64_t *s) {

	size_t i;
	uint64_t t[5], u[5], v, w;

	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1]  = ROTL64(s[6], 44);
		s[6]  = ROTL64(s[9], 20);
		s[9]  = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[2], 62);
		s[2]  = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19], 8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[4], 27);
		s[4]  = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21], 2);
		s[21] = ROTL64(s[8], 55);
		s[8]  = ROTL64(s[16], 45);
		s[16] = ROTL64(s[5], 36);
		s[5]  = ROTL64(s[3], 28);
		s[3]  = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[7], 6);
		s[7]  = ROTL64(s[10], 3);
		s[10] = ROTL64(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= host_keccak_round_constants[i];
	}
}

static __constant__ __align__(128) uint64_t c_state[25];
static __constant__ __align__(128) uint64_t c_message80_trail;

__host__
void cuda_base_keccak512_setBlock_80(void *endiandata) {

    uint64_t state[25];
    uint64_t *inMsg = (uint64_t*)endiandata;

    memcpy(state, inMsg, 72);
    memset(&state[9], 0, 128);

    keccak_block_host(state);

    cudaMemcpyToSymbol(c_state, state, sizeof(state), 0, cudaMemcpyHostToDevice);

    inMsg += 9;
    cudaMemcpyToSymbol(&c_message80_trail, inMsg, sizeof(c_message80_trail), 0, cudaMemcpyHostToDevice);
}


__global__
__launch_bounds__(KECCAK512_TPB80, 3)
void cuda_base_keccak512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t *g_hash) {

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

		uint2 keccak_gpu_state[25];

        for (int i = 0; i < 25; i++) {
            keccak_gpu_state[i] = vectorize(c_state[i]);
        }

        // sponge last 8 bytes and pad
        keccak_gpu_state[0] ^= vectorize(REPLACE_HIDWORD(c_message80_trail, cuda_swab32(startNonce + thread)));
		keccak_gpu_state[1] ^= vectorize(0x0000000000000001ULL);
		keccak_gpu_state[8] ^= vectorize(0x8000000000000000ULL);

		keccak_block(keccak_gpu_state);
		last_keccak_block(keccak_gpu_state);

        uint64_t *outHash = (uint64_t *)&g_hash[thread << 4];
		for(int i=0; i<8; i++) {
			outHash[i] = devectorize(keccak_gpu_state[i]);
		}
	}
}

__host__
void cuda_base_keccak512_cpu_hash_80(const uint32_t threads, const uint32_t startNonce,  uint32_t *d_hash) {

	const uint32_t threadsperblock = KECCAK512_TPB80;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_base_keccak512_gpu_hash_80<<<grid, block>>>(threads, startNonce, d_hash);

}

#pragma endregion

#pragma region Keccak512_64

__global__
__launch_bounds__(KECCAK512_TPB64, 3)
void cuda_base_keccak512_gpu_hash_64(const uint32_t threads, uint32_t *g_hash) {

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

        uint64_t *inpHash = (uint64_t *)&g_hash[thread << 4];
		uint2 keccak_gpu_state[25];

		for (int i = 0; i<8; i++) {
			keccak_gpu_state[i] = vectorize(inpHash[i]);
		}
		keccak_gpu_state[8] = vectorize(0x8000000000000001ULL);

		for (int i=9; i<25; i++) {
			keccak_gpu_state[i] = make_uint2(0, 0);
		}

		keccak_block(keccak_gpu_state);
		last_keccak_block(keccak_gpu_state);

		for(int i=0; i<8; i++) {
			inpHash[i] = devectorize(keccak_gpu_state[i]);
		}
	}
}

__host__
void cuda_base_keccak512_cpu_hash_64(const uint32_t threads,  uint32_t *d_hash) {

	const uint32_t threadsperblock = KECCAK512_TPB64;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_base_keccak512_gpu_hash_64<<<grid, block>>>(threads, d_hash);

}

#pragma endregion

#pragma region Keccak512_64_final

__global__
__launch_bounds__(KECCAK512_TPB64F, 3)
void cuda_base_keccak512_gpu_hash_64f(const uint32_t threads, const uint32_t* __restrict__ g_hash, const uint32_t startNonce, uint32_t *resNonce, const uint64_t target) {

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

        const uint64_t *inpHash = (uint64_t *)&g_hash[thread << 4];
		uint2 keccak_gpu_state[25];

		for (int i = 0; i<8; i++) {
			keccak_gpu_state[i] = vectorize(inpHash[i]);
		}
		keccak_gpu_state[8] = vectorize(0x8000000000000001ULL);

		for (int i=9; i<25; i++) {
			keccak_gpu_state[i] = make_uint2(0, 0);
		}

		keccak_block(keccak_gpu_state);
		last_keccak_block_final(keccak_gpu_state);

		if (devectorize(keccak_gpu_state[3]) <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], startNonce + thread);
            if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

__host__
void cuda_base_keccak512_cpu_hash_64f(const uint32_t threads, const uint32_t *d_hash, const uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target) {

	const uint32_t threadsperblock = KECCAK512_TPB64F;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_base_keccak512_gpu_hash_64f<<<grid, block>>>(threads, d_hash, startNonce, d_resNonce, target);

}

#pragma endregion

//__host__
//void keccak512_setBlock_80(int thr_id, uint32_t *endiandata)
//{
//	jackpot_keccak512_cpu_setBlock((void*)endiandata, 80);
//}
//
//__host__
//void keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
//{
//	jackpot_keccak512_cpu_hash(thr_id, threads, startNounce, d_hash, 0);
//}
