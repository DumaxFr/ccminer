#include "x11/cuda_x11.h"
#include "Algo512/cuda_b_cubehash512.h"
#include "Algo512/cuda_b_echo512.h"
#include "Algo512/scuda_b_jh512.h"
#include "cuda_phi2_skein512.h"

extern void cuda_phi2_lyra2_cpu_init(uint64_t *);
extern void cuda_phi2_lyra2_cpu_hash_32x2(const uint32_t, uint32_t*);

extern void cuda_phi2_branhc_streeb_echo512_cpu_hash_64(const uint32_t, uint32_t*);

