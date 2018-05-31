#include "x11/cuda_x11.h"
#include "Algo512\cuda_b_cubehash512.h"
#include "Algo512\cuda_b_echo512.h"
#include "Algo512\cuda_b_fugue512.h"

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int swap);
extern void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void streebog_hash_64_maxwell(int thr_id, uint32_t threads, uint32_t *d_hash);

//extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
//extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
//extern void x13_fugue512_cpu_free(int thr_id);

extern void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);

