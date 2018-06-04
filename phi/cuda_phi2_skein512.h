#ifndef CUDA_PHI2_SKEIN512
#define CUDA_PHI2_SKEIN512

#include <stdint.h>

void cuda_phi2_skein512_cpu_hash_64(const uint32_t threads, uint32_t *d_hash);
void cuda_phi2_skein512_cpu_hash_64f(const uint32_t threads, uint32_t * d_hash, const uint32_t startNonce, uint32_t * d_resNonce, const uint64_t target);

#endif // !CUDA_PHI2_SKEIN512


