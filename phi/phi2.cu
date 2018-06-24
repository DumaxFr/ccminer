//
//  PHI2 algo (with smart contracts header)
//  tpruvot May 2018
//  DumaxFr June 2018
//

extern "C" {
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
#include "sph/sph_jh.h"
#include "sph/sph_streebog.h"
#include "sph/sph_echo.h"
#include "sph/sph_skein.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include "cuda_phi2.h"

#include <stdio.h>
#include <memory.h>

static uint32_t* d_hash[MAX_GPUS];
static uint32_t* d_resNonce[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];

static bool has_roots;

extern "C" void phi2hash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[64] = { 0 };
    unsigned char hashA[64] = { 0 };
    unsigned char hashB[64] = { 0 };

    sph_cubehash512_context ctx_cubehash;
    sph_jh512_context ctx_jh;
    sph_gost512_context ctx_gost;
    sph_echo512_context ctx_echo;
    sph_skein512_context ctx_skein;

    sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)input, has_roots ? 144 : 80);
    sph_cubehash512_close(&ctx_cubehash, (void*)hashB);

    LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    sph_jh512_init(&ctx_jh);
    sph_jh512(&ctx_jh, (const void*)hashA, 64);
    sph_jh512_close(&ctx_jh, (void*)hash);

    if (hash[0] & 1) {
        sph_gost512_init(&ctx_gost);
        sph_gost512(&ctx_gost, (const void*)hash, 64);
        sph_gost512_close(&ctx_gost, (void*)hash);
    } else {
        sph_echo512_init(&ctx_echo);
        sph_echo512(&ctx_echo, (const void*)hash, 64);
        sph_echo512_close(&ctx_echo, (void*)hash);

        sph_echo512_init(&ctx_echo);
        sph_echo512(&ctx_echo, (const void*)hash, 64);
        sph_echo512_close(&ctx_echo, (void*)hash);
    }

    sph_skein512_init(&ctx_skein);
    sph_skein512(&ctx_skein, (const void*)hash, 64);
    sph_skein512_close(&ctx_skein, (void*)hash);

    for (int i=0; i<32; i++)
        hash[i] ^= hash[i+32];

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "phi2-"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

#ifdef _PROFILE_METRICS
#define _PROFILE_METRICS_PHI
#endif // _PROFILE_METRICS

#ifdef _PROFILE_METRICS_PHI
#define HASH_FUNC_COUNT 5
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


extern "C" int scanhash_phi2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 16 : 15;
	if (device_sm[dev_id] >= 600) intensity = 16;

    // least common multiple of all algo TPBs
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
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], MAX_NONCES * sizeof(uint32_t)));

		size_t matrix_sz = sizeof(uint64_t) * 4 * 4;
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
        cuda_phi2_lyra2_cpu_init(d_matrix[thr_id]);

		init[thr_id] = true;
	}

	has_roots = false;
	uint32_t endiandata[36];
	for (int k = 0; k < 36; k++) {
		be32enc(&endiandata[k], pdata[k]);
		if (k >= 20 && pdata[k]) has_roots = true;
	}

    cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));

    #ifdef _PROFILE_METRICS_PHI
    float milliseconds;
    bool metrics_do_first_start = false;
    START_METRICS
    #endif // _PROFILE_METRICS_PHI

	if (has_roots)
		cubehash512_setBlock_144(thr_id, endiandata);
	else
        cuda_base_cubehash512_setBlock_80(endiandata);


	do {

        #ifdef _PROFILE_METRICS_PHI
        if (metrics_do_first_start) {
            START_METRICS
        } else {
            metrics_do_first_start = true;
        }
        #endif // _PROFILE_METRICS_PHI
		if (has_roots)
			cubehash512_cuda_hash_144(thr_id, throughput, pdata[19], d_hash_512[thr_id]);
		else
            cuda_base_cubehash512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
        TRACE("CubeHash512-xxx: ")
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(0)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        cuda_phi2_lyra2_cpu_hash_32x2(throughput, d_hash[thr_id]);
        TRACE("Lyra2-64       : ")
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(1)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        cuda_base_jh512_cpu_hash_64(throughput, d_hash[thr_id]);
        TRACE("Jh512-64       : ")
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(2)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        cuda_phi2_branhc_streeb_echo512_cpu_hash_64(throughput, d_hash[thr_id]);
        TRACE("Brnch StreeEcho: ")
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(3)
        
        START_METRICS
        #endif // _PROFILE_METRICS_PHI
        // Phi2 skein512 and hash[i] ^= hash[i+32]
        cuda_phi2_skein512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);
        //cuda_phi2_skein512_cpu_hash_64(throughput, d_hash[thr_id]); // for debug with --cputest
        TRACE("skein512-64    : ")
        #ifdef _PROFILE_METRICS_PHI
        STOP_METRICS(4)
        #endif // _PROFILE_METRICS_PHI


        cudaMemcpy(work->nonces, d_resNonce[thr_id], MAX_NONCES * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (work->nonces[0] != UINT32_MAX) {
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			phi2hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
                *hashes_done = pdata[19] - first_nonce + throughput;
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					phi2hash(vhash, endiandata);
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
extern "C" void free_phi2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();
	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
