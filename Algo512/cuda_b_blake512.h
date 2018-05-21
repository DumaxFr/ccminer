#ifndef CUDA_BASE_BLAKE512
#define CUDA_BASE_BLAKE512

#include <stdint.h>

void cuda_base_blake512_cpu_setBlock_80(uint32_t *endiandata);
void cuda_base_blake512_cpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t *d_outputHash);

void cuda_base_blake512_cpu_hash_64(const uint32_t threads, uint32_t *d_outputHash);
void cuda_base_blake512_cpu_hash_64f(const uint32_t threads, const uint32_t *d_outputHash, const uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target);

#endif // !CUDA_BASE_BLAKE512
