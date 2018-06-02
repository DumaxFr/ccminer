#include "x11/cuda_x11.h"
#include "Algo512\cuda_b_cubehash512.h"
#include "Algo512\cuda_b_echo512.h"
#include "Algo512\cuda_b_fugue512.h"
#include "Algo512\cuda_b_jh512.h"
#include "Algo512\cuda_b_skein512.h"

extern void streebog_hash_64_maxwell(int thr_id, uint32_t threads, uint32_t *d_hash);

