//
//
//  PHI1612 algo
//  Skein + JH + CubeHash + Fugue + Gost + Echo
//
//  Implemented by anorganix @ bitcointalk on 01.10.2017
//  Feel free to send some satoshis to 1Bitcoin8tfbtGAQNFxDRUVUfFgFWKoWi9
//
//

extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_fugue.h"
#include "sph/sph_streebog.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "./cuda_phi.h"

#include <stdio.h>
#include <memory.h>

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

extern "C" void phihash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_jh512_context ctx_jh;
	sph_cubehash512_context ctx_cubehash;
	sph_fugue512_context ctx_fugue;
	sph_gost512_context ctx_gost;
	sph_echo512_context ctx_echo;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*)hash, 64);
	sph_gost512_close(&ctx_gost, (void*)hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	memcpy(output, hash, 32);
}

#define _DEBUG_PREFIX "phi"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };
//static bool use_compat_kernels[MAX_GPUS] = { 0 };

#ifdef _PROFILE_METRICS
#define _PROFILE_METRICS_PHI
#endif // _PROFILE_METRICS

#ifdef _PROFILE_METRICS_PHI
#define HASH_FUNC_COUNT 6
static float avgDuration64[HASH_FUNC_COUNT][MAX_GPUS] = { 0.0f };
static long totalRuns64[HASH_FUNC_COUNT][MAX_GPUS] = { 0l };

static cudaEvent_t phi_kernel_start[MAX_GPUS];
static cudaEvent_t phi_kernel_stop[MAX_GPUS];

#define START_METRICS { \
    if (opt_debug) { \
        milliseconds = 0.0; \
        cudaEventCreate(&phi_kernel_start[thr_id]); \
        cudaEventCreate(&phi_kernel_stop[thr_id]); \
        cudaEventRecord(phi_kernel_start[thr_id]); \
    } \
}

#define STOP_METRICS(kid) { \
    if (opt_debug) { \
        cudaEventRecord(phi_kernel_stop[thr_id]); \
        cudaEventSynchronize(phi_kernel_stop[thr_id]); \
        cudaEventElapsedTime(&milliseconds, phi_kernel_start[thr_id], phi_kernel_stop[thr_id]); \
        cudaEventDestroy(phi_kernel_start[thr_id]); \
        cudaEventDestroy(phi_kernel_stop[thr_id]); \
        avgDuration64[kid][thr_id] += (milliseconds - avgDuration64[kid][thr_id]) / (float)(totalRuns64[kid][thr_id] + 1); \
        totalRuns64[kid][thr_id]++; \
    } \
}

#define PRINT_METRICS { \
    if (opt_debug) { \
        for (int i = 0; i < HASH_FUNC_COUNT; i++) { \
            gpulog(LOG_BLUE, thr_id, "%02d-64 AvgDuration after %d runs : %f ms", \
                i, totalRuns64[i][thr_id], avgDuration64[i][thr_id]); \
        } \
    } \
}

#endif // _PROFILE_METRICS_PHI


extern "C" int scanhash_phi(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 19 : 18; // 2^18 = 262144 cuda threads
	if (device_sm[dev_id] >= 600) intensity = 20;

    // least common multiple of all TPB algos
    uint32_t lcm = 7680; // should be calculated by a cuda_get_lcm function
    uint32_t throughput = cuda_default_throughput_lcm(thr_id, 1U << intensity, lcm);
    if (init[thr_id] && max_nonce - first_nonce < throughput) {
        throughput = max_nonce - first_nonce + lcm - ((max_nonce - first_nonce) % lcm);
    }

	if (opt_benchmark)
		ptarget[7] = 0xf;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		quark_skein512_cpu_init(thr_id, throughput);
        cuda_base_echo512_cpu_init(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput), -1);
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

    cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));

    #ifdef _PROFILE_METRICS_PHI
    float milliseconds;
    boolean metrics_do_first_start = false;
    START_METRICS
    #endif // _PROFILE_METRICS_PHI

	cuda_base_skein512_setBlock_80((void*)endiandata);

	do {

        // Hash with CUDA
        #ifdef _PROFILE_METRICS_PHI
        if (metrics_do_first_start) {
            START_METRICS
        } else {
            metrics_do_first_start = true;
        }
        #endif // _PROFILE_METRICS_PHI
		cuda_base_skein512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(0)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        cuda_base_jh512_cpu_hash_64(throughput, d_hash[thr_id]);
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(1)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        cuda_base_cubehash512_cpu_hash_64(throughput, d_hash[thr_id]);
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(2)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        cuda_base_fugue512_cpu_hash_64(throughput, d_hash[thr_id]);
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(3)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
		streebog_hash_64_maxwell(thr_id, throughput, d_hash[thr_id]);
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(4)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        cuda_base_echo512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(5)
        #endif // _PROFILE_METRICS_PHI

        *hashes_done = pdata[19] - first_nonce + throughput;

        cudaMemcpy(work->nonces, d_resNonce[thr_id], MAX_NONCES * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (work->nonces[0] != UINT32_MAX) {
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			phihash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					phihash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
                    pdata[19] = (uint64_t)throughput + pdata[19] >= max_nonce ? max_nonce : pdata[19] + throughput; 
				}
				return work->valid_nonces;
			} else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				cudaMemset(d_resNonce[thr_id], 0xFF, MAX_NONCES * sizeof(uint32_t));
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

    #ifdef _PROFILE_METRICS_PHI
    PRINT_METRICS
    #endif // _PROFILE_METRICS_PHI

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

#ifdef _PROFILE_METRICS_PHI
#undef START_METRICS
#undef STOP_METRICS
#undef PRINT_METRICS
#endif // _PROFILE_METRICS_PHI


// cleanup
extern "C" void free_phi(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();
	cudaFree(d_hash[thr_id]);
	//cudaFree(d_resNonce[thr_id]); // done in cuda_chech_cpu_free
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
