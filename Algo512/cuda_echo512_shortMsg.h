#ifndef CUDA_BASE_ECHO512
#define CUDA_BASE_ECHO512

#include <stdint.h>

void echo512_cpu_hash_64(const uint32_t threads, uint32_t *d_hash);

void echo512_cpu_init(int thr_id);
void echo512_setBlock_80(void *pdata);
void echo512_cuda_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

#endif // !CUDA_BASE_ECHO512
