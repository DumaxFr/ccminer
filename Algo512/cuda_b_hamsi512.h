#ifndef CUDA_BASE_HAMSI512
#define CUDA_BASE_HAMSI512

#include <stdint.h>

void cuda_base_hamsi512_cpu_init();

void cuda_base_hamsi512_setBlock_80(void * endiandata);
void cuda_base_hamsi512_cpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t * d_hash);

void cuda_base_hamsi512_cpu_hash_64(const uint32_t threads, uint32_t * d_hash);
void cuda_base_hamsi512_cpu_hash_64f(const uint32_t threads, const uint32_t * d_hash, const uint32_t startNonce, uint32_t * d_resNonce, const uint64_t target);

#endif // !CUDA_BASE_HAMSI512
