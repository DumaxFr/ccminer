#include "x11/cuda_x11.h"

#include "Algo512/cuda_b_blake512.h"
#include "Algo512/cuda_b_bmw512.h"
#include "Algo512/cuda_b_cubehash512.h"
#include "Algo512/cuda_b_echo512.h"
#include "Algo512/cuda_b_fugue512.h"
#include "Algo512/cuda_b_hamsi512.h"
#include "Algo512/cuda_b_jh512.h"
#include "Algo512/cuda_b_keccak512.h"
#include "Algo512/cuda_b_luffa512.h"
#include "Algo512/cuda_b_sha512.h"
#include "Algo512/cuda_b_shabal512.h"
#include "Algo512/cuda_b_shavite512.h"
#include "Algo512/cuda_b_skein512.h"
#include "Algo512/cuda_b_whirlpool.h"

//extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
//extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, const int outlen);

// ---- 80 bytes kernels

void quark_bmw512_cpu_setBlock_80(void *pdata);
void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);

void groestl512_setBlock_80(int thr_id, uint32_t *endiandata);
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void keccak512_setBlock_80(int thr_id, uint32_t *endiandata);
void keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void cubehash512_setBlock_80(int thr_id, uint32_t* endiandata);
void cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_simd512_setBlock_80(void *pdata);
void x16_simd512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_whirlpool512_init(int thr_id, uint32_t threads);
void x16_whirlpool512_setBlock_80(void* endiandata);
void x16_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
