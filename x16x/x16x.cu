/**
 * X16R algorithm (X16 with Randomized chain order)
 *
 * tpruvot 2018 - GPL code
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

extern "C" {
    #include "sph/sph_blake.h"
    #include "sph/sph_bmw.h"
    #include "sph/sph_groestl.h"
    #include "sph/sph_skein.h"
    #include "sph/sph_jh.h"
    #include "sph/sph_keccak.h"

    #include "sph/sph_luffa.h"
    #include "sph/sph_cubehash.h"
    #include "sph/sph_shavite.h"
    #include "sph/sph_simd.h"
    #include "sph/sph_echo.h"

    #include "sph/sph_hamsi.h"
    #include "sph/sph_fugue.h"
    #include "sph/sph_shabal.h"
    #include "sph/sph_whirlpool.h"
    #include "sph/sph_sha2.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include "./cuda_x16x.h"

static uint32_t* d_hash[MAX_GPUS];
static uint32_t *d_x16ResNonce[MAX_GPUS];

enum Algo {
    BLAKE = 0,
    BMW,
    GROESTL,
    JH,
    KECCAK,
    SKEIN,
    LUFFA,
    CUBEHASH,
    SHAVITE,
    SIMD,
    ECHO,
    HAMSI,
    FUGUE,
    SHABAL,
    WHIRLPOOL,
    SHA512,
    HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
    "blake",
    "bmw512",
    "groestl",
    "jh512",
    "keccak",
    "skein",
    "luffa",
    "cube",
    "shavite",
    "simd",
    "echo",
    "hamsi",
    "fugue",
    "shabal",
    "whirlpool",
    "sha512",
    NULL
};

static __thread uint32_t s_ntime = UINT32_MAX;
static __thread bool s_implemented = false;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

static void getAlgoString16r(const uint32_t* prevblock, char *output) {
    char *sptr = output;
    uint8_t* data = (uint8_t*)prevblock;

    for (uint8_t j = 0; j < HASH_FUNC_COUNT; j++) {
        uint8_t b = (15 - j) >> 1; // 16 ascii hex chars, reversed
        uint8_t algoDigit = (j & 1) ? data[b] & 0xF : data[b] >> 4;
        if (algoDigit >= 10)
            sprintf(sptr, "%c", 'A' + (algoDigit - 10));
        else
            sprintf(sptr, "%u", (uint32_t)algoDigit);
        sptr++;
    }
    *sptr = '\0';
}

static void getAlgoString16s(const uint32_t* prevblock, char *output) {

    char *ptrHash = output;
    uint8_t* data = (uint8_t*)prevblock;

    strcpy(ptrHash, "0123456789ABCDEF");

    for (int j = 0; j < HASH_FUNC_COUNT; j++) {
        uint8_t b = (15 - j) >> 1;
        uint8_t algoDigit = (j & 1) ? data[b] & 0xF : data[b] >> 4;
        if (algoDigit > 0) {
            char pull = ptrHash[algoDigit];
            for (int i = algoDigit; i > 0; i--) {
                ptrHash[i] = ptrHash[i-1];
            }
            ptrHash[0] = pull;
        }
    }

}

static void getAlgoString(const uint32_t* prevblock, char *output, const char baseAlgo) {
    if (baseAlgo == 'r')
        getAlgoString16r(prevblock, output);
    else
        getAlgoString16s(prevblock, output);
}

// X16x CPU Hash (Validation)
extern "C" void x16x_hash(void *output, const void *input, const char variation) {
    unsigned char _ALIGN(64) hash[64];

    sph_blake512_context ctx_blake;
    sph_bmw512_context ctx_bmw;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    sph_luffa512_context ctx_luffa;
    sph_cubehash512_context ctx_cubehash;
    sph_shavite512_context ctx_shavite;
    sph_simd512_context ctx_simd;
    sph_echo512_context ctx_echo;
    sph_hamsi512_context ctx_hamsi;
    sph_fugue512_context ctx_fugue;
    sph_shabal512_context ctx_shabal;
    sph_whirlpool_context ctx_whirlpool;
    sph_sha512_context ctx_sha512;

    void *in = (void*)input;
    int size = 80;

    uint32_t *in32 = (uint32_t*)input;
    getAlgoString(&in32[1], hashOrder, variation);

    for (int i = 0; i < 16; i++) {
        const char elem = hashOrder[i];
        const uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

        switch (algo) {
            case BLAKE:
                sph_blake512_init(&ctx_blake);
                sph_blake512(&ctx_blake, in, size);
                sph_blake512_close(&ctx_blake, hash);
                break;
            case BMW:
                sph_bmw512_init(&ctx_bmw);
                sph_bmw512(&ctx_bmw, in, size);
                sph_bmw512_close(&ctx_bmw, hash);
                break;
            case GROESTL:
                sph_groestl512_init(&ctx_groestl);
                sph_groestl512(&ctx_groestl, in, size);
                sph_groestl512_close(&ctx_groestl, hash);
                break;
            case SKEIN:
                sph_skein512_init(&ctx_skein);
                sph_skein512(&ctx_skein, in, size);
                sph_skein512_close(&ctx_skein, hash);
                break;
            case JH:
                sph_jh512_init(&ctx_jh);
                sph_jh512(&ctx_jh, in, size);
                sph_jh512_close(&ctx_jh, hash);
                break;
            case KECCAK:
                sph_keccak512_init(&ctx_keccak);
                sph_keccak512(&ctx_keccak, in, size);
                sph_keccak512_close(&ctx_keccak, hash);
                break;
            case LUFFA:
                sph_luffa512_init(&ctx_luffa);
                sph_luffa512(&ctx_luffa, in, size);
                sph_luffa512_close(&ctx_luffa, hash);
                break;
            case CUBEHASH:
                sph_cubehash512_init(&ctx_cubehash);
                sph_cubehash512(&ctx_cubehash, in, size);
                sph_cubehash512_close(&ctx_cubehash, hash);
                break;
            case SHAVITE:
                sph_shavite512_init(&ctx_shavite);
                sph_shavite512(&ctx_shavite, in, size);
                sph_shavite512_close(&ctx_shavite, hash);
                break;
            case SIMD:
                sph_simd512_init(&ctx_simd);
                sph_simd512(&ctx_simd, in, size);
                sph_simd512_close(&ctx_simd, hash);
                break;
            case ECHO:
                sph_echo512_init(&ctx_echo);
                sph_echo512(&ctx_echo, in, size);
                sph_echo512_close(&ctx_echo, hash);
                break;
            case HAMSI:
                sph_hamsi512_init(&ctx_hamsi);
                sph_hamsi512(&ctx_hamsi, in, size);
                sph_hamsi512_close(&ctx_hamsi, hash);
                break;
            case FUGUE:
                sph_fugue512_init(&ctx_fugue);
                sph_fugue512(&ctx_fugue, in, size);
                sph_fugue512_close(&ctx_fugue, hash);
                break;
            case SHABAL:
                sph_shabal512_init(&ctx_shabal);
                sph_shabal512(&ctx_shabal, in, size);
                sph_shabal512_close(&ctx_shabal, hash);
                break;
            case WHIRLPOOL:
                sph_whirlpool_init(&ctx_whirlpool);
                sph_whirlpool(&ctx_whirlpool, in, size);
                sph_whirlpool_close(&ctx_whirlpool, hash);
                break;
            case SHA512:
                sph_sha512_init(&ctx_sha512);
                sph_sha512(&ctx_sha512, (const void*)in, size);
                sph_sha512_close(&ctx_sha512, (void*)hash);
                break;
        }
        in = (void*)hash;
        size = 64;
    }
    memcpy(output, hash, 32);
}

void whirlpool_midstate(void *state, const void *input) {
    sph_whirlpool_context ctx;

    sph_whirlpool_init(&ctx);
    sph_whirlpool(&ctx, input, 64);

    memcpy(state, ctx.state, 64);
}

static bool init[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "x16-"
#include "cuda_debug.cuh"

#ifdef _PROFILE_METRICS
#define _PROFILE_METRICS_X16
#endif // _PROFILE_METRICS

#ifdef _PROFILE_METRICS_X16
#define HASH_FUNC_COUNT 16
static float avgDuration80[HASH_FUNC_COUNT][MAX_GPUS] = { 0.0f };
static long totalRuns80[HASH_FUNC_COUNT][MAX_GPUS] = { 0l };
static float avgDuration64[HASH_FUNC_COUNT][MAX_GPUS] = { 0.0f };
static long totalRuns64[HASH_FUNC_COUNT][MAX_GPUS] = { 0l };
static float avgDuration64f[HASH_FUNC_COUNT][MAX_GPUS] = { 0.0f };
static long totalRuns64f[HASH_FUNC_COUNT][MAX_GPUS] = { 0l };
static float avgDurationCH[MAX_GPUS] = { 0.0f };
static long totalRunsCH[MAX_GPUS] = { 0l };
static float avgDurationGN[MAX_GPUS] = { 0.0f };
static long totalRunsGN[MAX_GPUS] = { 0l };

static cudaEvent_t x16_kernel_start[MAX_GPUS];
static cudaEvent_t x16_kernel_stop[MAX_GPUS];

static int algo80_foundShare[HASH_FUNC_COUNT][MAX_GPUS] = { 0 };
static int algo64_foundShare[HASH_FUNC_COUNT][MAX_GPUS] = { 0 };
static int algo64f_foundShare[HASH_FUNC_COUNT][MAX_GPUS] = { 0 };
static int algo80_fails[HASH_FUNC_COUNT][MAX_GPUS] = { 0 };
static int algo64_fails[HASH_FUNC_COUNT][MAX_GPUS] = { 0 };
static int algo64f_fails[HASH_FUNC_COUNT][MAX_GPUS] = { 0 };

#define START_METRICS { \
    if (opt_debug) { \
        milliseconds = 0.0; \
        cudaEventCreate(&x16_kernel_start[thr_id]); \
        cudaEventCreate(&x16_kernel_stop[thr_id]); \
        cudaEventRecord(x16_kernel_start[thr_id]); \
    } \
}

#define STOP_METRICS_80(kid) { \
    if (opt_debug) { \
        cudaEventRecord(x16_kernel_stop[thr_id]); \
        cudaEventSynchronize(x16_kernel_stop[thr_id]); \
        cudaEventElapsedTime(&milliseconds, x16_kernel_start[thr_id], x16_kernel_stop[thr_id]); \
        cudaEventDestroy(x16_kernel_start[thr_id]); \
        cudaEventDestroy(x16_kernel_stop[thr_id]); \
        avgDuration80[kid][thr_id] += (milliseconds - avgDuration80[kid][thr_id]) / (float)(totalRuns80[kid][thr_id] + 1); \
        totalRuns80[kid][thr_id]++; \
    } \
}

#define STOP_METRICS_64(kid) { \
    if (opt_debug) { \
        cudaEventRecord(x16_kernel_stop[thr_id]); \
        cudaEventSynchronize(x16_kernel_stop[thr_id]); \
        cudaEventElapsedTime(&milliseconds, x16_kernel_start[thr_id], x16_kernel_stop[thr_id]); \
        cudaEventDestroy(x16_kernel_start[thr_id]); \
        cudaEventDestroy(x16_kernel_stop[thr_id]); \
        if (cudaHashFinalDone) { \
            avgDuration64f[kid][thr_id] += (milliseconds - avgDuration64f[kid][thr_id]) / (float)(totalRuns64f[kid][thr_id] + 1); \
            totalRuns64f[kid][thr_id]++; \
        } else { \
            avgDuration64[kid][thr_id] += (milliseconds - avgDuration64[kid][thr_id]) / (float)(totalRuns64[kid][thr_id] + 1); \
            totalRuns64[kid][thr_id]++; \
        } \
    } \
}

#define STOP_METRICS_CH { \
    if (opt_debug) { \
        cudaEventRecord(x16_kernel_stop[thr_id]); \
        cudaEventSynchronize(x16_kernel_stop[thr_id]); \
        cudaEventElapsedTime(&milliseconds, x16_kernel_start[thr_id], x16_kernel_stop[thr_id]); \
        cudaEventDestroy(x16_kernel_start[thr_id]); \
        cudaEventDestroy(x16_kernel_stop[thr_id]); \
        avgDurationCH[thr_id] += (milliseconds - avgDurationCH[thr_id]) / (float)(totalRunsCH[thr_id] + 1); \
        totalRunsCH[thr_id]++; \
    } \
}

#define STOP_METRICS_GN { \
    if (opt_debug) { \
        cudaEventRecord(x16_kernel_stop[thr_id]); \
        cudaEventSynchronize(x16_kernel_stop[thr_id]); \
        cudaEventElapsedTime(&milliseconds, x16_kernel_start[thr_id], x16_kernel_stop[thr_id]); \
        cudaEventDestroy(x16_kernel_start[thr_id]); \
        cudaEventDestroy(x16_kernel_stop[thr_id]); \
        avgDurationGN[thr_id] += (milliseconds - avgDurationGN[thr_id]) / (float)(totalRunsGN[thr_id] + 1); \
        totalRunsGN[thr_id]++; \
    } \
}

#define PRINT_METRICS { \
    if (opt_debug) { \
        for (int i = 0; i < HASH_FUNC_COUNT; i++) { \
            int logColor = algo80_foundShare[i][thr_id] > 0 ? LOG_BLUE : LOG_DEBUG; \
            gpulog(logColor, thr_id, "%9s-80  AvgDuration after %6d runs : %7.4f ms f:%5d r:%d", \
                algo_strings[i], totalRuns80[i][thr_id], avgDuration80[i][thr_id], algo80_foundShare[i][thr_id], algo80_fails[i][thr_id]); \
            logColor = algo64_foundShare[i][thr_id] > 0 ? LOG_BLUE : LOG_DEBUG; \
            gpulog(logColor, thr_id, "%9s-64  AvgDuration after %6d runs : %7.4f ms f:%5d r:%d", \
                algo_strings[i], totalRuns64[i][thr_id], avgDuration64[i][thr_id], algo64_foundShare[i][thr_id], algo64_fails[i][thr_id]); \
            if (totalRuns64f[i][thr_id] != 0) { \
                logColor = algo64f_foundShare[i][thr_id] > 0 ? LOG_WARNING : LOG_DEBUG; \
                gpulog(logColor, thr_id, "%9s-64f AvgDuration after %6d runs : %7.4f ms f:%5d r:%d", \
                    algo_strings[i], totalRuns64f[i][thr_id], avgDuration64f[i][thr_id], algo64f_foundShare[i][thr_id], algo64f_fails[i][thr_id]); \
            } \
        } \
    } \
}

#define COUNT_FOUND { \
    algo80_foundShare[algo80][thr_id]++; \
    char elem; \
    uint8_t algo64; \
    bool algo64_participate[HASH_FUNC_COUNT] = {false}; \
    for (int i = 1; i < 15; i++) { \
        elem = hashOrder[i]; \
        algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0'; \
        algo64_participate[algo64] = true; \
    } \
    elem = hashOrder[15]; \
    algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0'; \
    if (cudaHashFinalDone) { \
        algo64f_foundShare[algo64][thr_id]++; \
    } else { \
        algo64_participate[algo64] = true; \
    } \
    for (int i = 0; i < HASH_FUNC_COUNT; i++) { \
        if (algo64_participate[i]) algo64_foundShare[i][thr_id]++; \
    } \
}

#define COUNT_FAILS { \
    algo80_fails[algo80][thr_id]++; \
    char elem; \
    uint8_t algo64; \
    bool algo64_participate[HASH_FUNC_COUNT] = {false}; \
    for (int i = 1; i < 15; i++) { \
        elem = hashOrder[i]; \
        algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0'; \
        algo64_participate[algo64] = true; \
    } \
    elem = hashOrder[15]; \
    algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0'; \
    if (cudaHashFinalDone) { \
        algo64f_fails[algo64][thr_id]++; \
    } else { \
        algo64_participate[algo64] = true; \
    } \
    for (int i = 0; i < HASH_FUNC_COUNT; i++) { \
        if (algo64_participate[i]) algo64_fails[i][thr_id]++; \
    } \
}

#endif // _PROFILE_METRICS_X16


extern "C" int scanhash_x16x(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, const char variation) {
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    const int dev_id = device_map[thr_id];
    int intensity = (device_sm[dev_id] > 600 && !is_windows()) ? 20 : 19;
    if (strstr(device_name[dev_id], "GTX 1080")) intensity = 20;

    uint32_t lcm = 15360; // should be calculated by a cuda_get_lcm function
    uint32_t throughput = cuda_default_throughput_lcm(thr_id, 1U << intensity, lcm);
    if (init[thr_id] && max_nonce - first_nonce < throughput) {
        throughput = max_nonce - first_nonce + lcm - ((max_nonce - first_nonce) % lcm);
    }

    if (!init[thr_id]) {
        cudaSetDevice(device_map[thr_id]);
        if (opt_cudaschedule == -1 && gpu_threads == 1) {
            cudaDeviceReset();
            // reduce cpu usage
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        }
        gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

        quark_blake512_cpu_init(thr_id, throughput);
        quark_groestl512_cpu_init(thr_id, throughput);
        quark_keccak512_cpu_init(thr_id, throughput);
        cuda_base_keccak512_cpu_init();
        x11_simd512_cpu_init(thr_id, throughput); // 64
        cuda_base_echo512_cpu_init(thr_id);
        cuda_base_hamsi512_cpu_init();
        //x15_whirlpool_cpu_init(thr_id, throughput, 0);
        cuda_base_whirlpool_cpu_init();
        x16_whirlpool512_init(thr_id, throughput); // 80
        cuda_base_sha512_cpu_init();

        CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput), 0);
 		CUDA_SAFE_CALL(cudaMalloc(&d_x16ResNonce[thr_id], MAX_NONCES * sizeof(uint32_t)));

        cuda_check_cpu_init(thr_id, throughput);

        cudaDeviceSynchronize();

        init[thr_id] = true;

		if (opt_debug)
			gpulog(LOG_DEBUG, thr_id, "X16%c init done", variation);

    }

    if (opt_benchmark) {
        ((uint32_t*)ptarget)[7] = 0x0;
		((uint32_t*)pdata)[1] = 0x01234567;
		((uint32_t*)pdata)[2] = 0x89ABCDEF;
    }
    uint32_t _ALIGN(64) endiandata[20];

    for (int k = 0; k < 19; k++)
        be32enc(&endiandata[k], pdata[k]);

    cudaMemset(d_x16ResNonce[thr_id], 0xff, MAX_NONCES * sizeof(uint32_t));
    
    uint32_t ntime = swab32(pdata[17]);
    if (s_ntime != ntime) {
        getAlgoString(&endiandata[1], hashOrder, variation);
        s_ntime = ntime;
        s_implemented = true;
        if (opt_debug && !thr_id) {
            applog(LOG_DEBUG, "hash order %s (%08x)", hashOrder, ntime);
        }
    }

    if (!s_implemented) {
        sleep(1);
        return -1;
    }

    cuda_check_cpu_setTarget(ptarget);

    char elem = hashOrder[0];
    const uint8_t algo80 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

    #ifdef _PROFILE_METRICS_X16
    float milliseconds;
    bool metrics_do_first_start = false;
    START_METRICS
    #endif // _PROFILE_METRICS_X16

    switch (algo80) {
        case BLAKE:
            cuda_base_blake512_cpu_setBlock_80(endiandata);
            break;
        case BMW:
            cuda_base_bmw512_cpu_setBlock_80(endiandata);
            break;
        case GROESTL:
            groestl512_setBlock_80(thr_id, endiandata);
            break;
        case JH:
            // !!!
            jh512_setBlock_80(thr_id, endiandata);
            break;
        case KECCAK:
            // !!!
            keccak512_setBlock_80(thr_id, endiandata);
            //cuda_base_keccak512_setBlock_80(endiandata);
            break;
        case SKEIN:
            cuda_base_skein512_setBlock_80((void*)endiandata);
            break;
        case LUFFA:
            cuda_base_luffa512_setBlock_80((void*)endiandata);
            break;
        case CUBEHASH:
            cuda_base_cubehash512_setBlock_80(endiandata);
            break;
        case SHAVITE:
            cuda_base_shavite512_setBlock_80((void*)endiandata);
            break;
        case SIMD:
            x16_simd512_setBlock_80((void*)endiandata);
            break;
        case ECHO:
            cuda_base_echo512_setBlock_80((void*)endiandata);
            break;
        case HAMSI:
            cuda_base_hamsi512_setBlock_80((void*)endiandata);
            break;
        case FUGUE:
            cuda_base_fugue512_setBlock_80((void*)pdata);
            break;
        case SHABAL:
            cuda_base_shabal512_setBlock_80((void*)endiandata);
            break;
        case WHIRLPOOL:
            x16_whirlpool512_setBlock_80((void*)endiandata);
            break;
        case SHA512:
            cuda_base_sha512_setBlock_80(endiandata);
            break;
        default:
            {
                if (!thr_id)
                    applog(LOG_WARNING, "kernel %s %c unimplemented, order %s", algo_strings[algo80], elem, hashOrder);
                s_implemented = false;
                sleep(5);
                return -1;
            }
    }

    int warn = 0;

    do {
        // Hash with CUDA

        bool cudaHashFinalDone = false;

        #ifdef _PROFILE_METRICS_X16
        if (metrics_do_first_start) {
            START_METRICS
        } else {
            metrics_do_first_start = true;
        }
        #endif

        switch (algo80) {
            case BLAKE:
                cuda_base_blake512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("blake80:");
                break;
            case BMW:
                cuda_base_bmw512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("bmw80  :");
                break;
            case GROESTL:
                groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
                TRACE("grstl80:");
                break;
            case JH:
                // !!!
                jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
                TRACE("jh51280:");
                break;
            case KECCAK:
                // !!!
                keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
                //cuda_base_keccak512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("kecck80:");
                break;
            case SKEIN:
                cuda_base_skein512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("skein80:");
                break;
            case LUFFA:
                cuda_base_luffa512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("luffa80:");
                break;
            case CUBEHASH:
                cuda_base_cubehash512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("cube 80:");
                break;
            case SHAVITE:
                cuda_base_shavite512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("shavite:");
                break;
            case SIMD:
                x16_simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
                TRACE("simd512:");
                break;
            case ECHO:
                cuda_base_echo512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("echo   :");
                break;
            case HAMSI:
                cuda_base_hamsi512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("hamsi  :");
                break;
            case FUGUE:
                cuda_base_fugue512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("fugue  :");
                break;
            case SHABAL:
                cuda_base_shabal512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("shabal :");
                break;
            case WHIRLPOOL:
                x16_whirlpool512_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
                TRACE("whirl  :");
                break;
            case SHA512:
                cuda_base_sha512_cuda_hash_80(throughput, pdata[19], d_hash[thr_id]);
                TRACE("sha512 :");
                break;
        }

        #ifdef _PROFILE_METRICS_X16
        STOP_METRICS_80(algo80)
        #endif // _PROFILE_METRICS_X16

        for (int i = 1; i < 16; i++) {
            const char elem = hashOrder[i];
            const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

            #ifdef _PROFILE_METRICS_X16
            START_METRICS
            #endif // _PROFILE_METRICS_X16

            switch (algo64) {
                case BLAKE:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_blake512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_blake512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("blake  :");
                    break;
                case BMW:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_bmw512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_bmw512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("bmw    :");
                    break;
                case GROESTL:
                    quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
                    TRACE("groestl:");
                    break;
                case JH:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_jh512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_jh512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("jh512  :");
                    break;
                case KECCAK:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_keccak512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_keccak512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("keccak :");
                    break;
                case SKEIN:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_skein512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_skein512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("skein  :");
                    break;
                case LUFFA:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_luffa512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_luffa512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("luffa  :");
                    break;
                case CUBEHASH:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_cubehash512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_cubehash512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("cube   :");
                    break;
                case SHAVITE:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_shavite512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_shavite512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("shavite:");
                    break;
                case SIMD:
                    x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
                    TRACE("simd   :");
                    break;
                case ECHO:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_echo512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_echo512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("echo   :");
                    break;
                case HAMSI:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_hamsi512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_hamsi512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("hamsi  :");
                    break;
                case FUGUE:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_fugue512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_fugue512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("fugue  :");
                    break;
                case SHABAL:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_shabal512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_shabal512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("shabal :");
                    break;
                case WHIRLPOOL:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_whirlpool_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_whirlpool_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("whirlpool:");
                    break;
                case SHA512:
                    if (i == 15) {
                        cudaHashFinalDone = true;
                        cuda_base_sha512_cpu_hash_64f(throughput, d_hash[thr_id], pdata[19], d_x16ResNonce[thr_id], *(uint64_t*)&ptarget[6]);
                    } else {
                        cuda_base_sha512_cpu_hash_64(throughput, d_hash[thr_id]);
                    }
                    TRACE("sha512 :");
                    break;
            }
        
            #ifdef _PROFILE_METRICS_X16
            STOP_METRICS_64(algo64)
            #endif // _PROFILE_METRICS_X16
        }

        *hashes_done = pdata[19] - first_nonce + throughput;

        if (cudaHashFinalDone) {
            #ifdef _PROFILE_METRICS_X16
            START_METRICS
            #endif // _PROFILE_METRICS_X16
            cudaMemcpy(work->nonces, d_x16ResNonce[thr_id], MAX_NONCES * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            // stay compliant with cuda_check_hash_suppl
            if (work->nonces[1] == UINT32_MAX)
                work->nonces[1] = 0;
            #ifdef _PROFILE_METRICS_X16
            STOP_METRICS_GN
            #endif // _PROFILE_METRICS_X16
        } else {
            #ifdef _PROFILE_METRICS_X16
            START_METRICS
            #endif // _PROFILE_METRICS_X16
            work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
            #ifdef _PROFILE_METRICS_X16
            STOP_METRICS_CH
            #endif // _PROFILE_METRICS_X16
        }


        if (work->nonces[0] != UINT32_MAX) {
            const uint32_t Htarg = ptarget[7];
            uint32_t _ALIGN(64) vhash[8];
            be32enc(&endiandata[19], work->nonces[0]);
            x16x_hash(vhash, endiandata, variation);

            if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                #ifdef _PROFILE_METRICS_X16
                COUNT_FOUND
                #endif // _PROFILE_METRICS_X16
                work->valid_nonces = 1;
                if (!cudaHashFinalDone)
                    work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
                work_set_target_ratio(work, vhash);
                if (work->nonces[1] != 0) {
                    be32enc(&endiandata[19], work->nonces[1]);
                    x16x_hash(vhash, endiandata, variation);
                    bn_set_target_ratio(work, vhash, 1);
                    work->valid_nonces++;
                    // maybe a third+ valid nonce in this grid
                    pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
                } else {
                    // don't see the reason to hash again from nonce+1 to startNonce+throughput
                    // next startNonce should be startNonce+throughput
                    pdata[19] = (uint64_t)throughput + pdata[19] >= max_nonce ? max_nonce : pdata[19] + throughput; 
                    //pdata[19] = work->nonces[0] + 1; // cursor
                }
                return work->valid_nonces;
            //} else if (Htarg == vhash[7] && ptarget[6] == vhash[6]) {
            //    gpulog(LOG_NOTICE, thr_id, "fast check : %08x is a false positive. %s", work->nonces[0], hashOrder);
            } else {
                gpu_increment_reject(thr_id);
                #ifdef _PROFILE_METRICS_X16
                COUNT_FAILS
                #endif // _PROFILE_METRICS_X16
                if (!warn) {
                    if (opt_debug) {
                        gpulog(LOG_DEBUG, thr_id, "result for %08x does not validate on CPU! %s", work->nonces[0], hashOrder);
                        //gpulog(LOG_DEBUG, thr_id, "target short : %64s", bin2hex((uchar *)&ptarget[6], 8));
                        //gpulog(LOG_DEBUG, thr_id, "target long  : %s", bin2hex((uchar *)ptarget, 32));
                        //gpulog(LOG_DEBUG, thr_id, "cpuhash      : %s", bin2hex((uchar *)vhash, 32));
                        //gpulog(LOG_DEBUG, thr_id, "data 00..04 : %s", bin2hex((uchar *)pdata, 20));
                        //gpulog(LOG_DEBUG, thr_id, "data 05..09 : %s", bin2hex((uchar *)&pdata[5], 20));
                        //gpulog(LOG_DEBUG, thr_id, "data 10..14 : %s", bin2hex((uchar *)&pdata[10], 20));
                        //gpulog(LOG_DEBUG, thr_id, "data 15..19 : %s", bin2hex((uchar *)&pdata[15], 20));
                        //gpulog(LOG_DEBUG, thr_id, "data 00     : %08x", pdata[0]);
                    }
                    warn++;
                    // don't see the reason to hash again from nonce to startNonce+throughput
                    //pdata[19] = work->nonces[0] + 1;
                    //continue;
                } else {
                    if (!opt_quiet) {
                        gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
                            work->nonces[0], algo_strings[algo80], hashOrder);
                        //if (opt_debug) {
                        //    gpulog(LOG_DEBUG, thr_id, "target short : %64s", bin2hex((uchar *)&ptarget[6], 8));
                        //    gpulog(LOG_DEBUG, thr_id, "target long  : %s", bin2hex((uchar *)ptarget, 32));
                        //    gpulog(LOG_DEBUG, thr_id, "cpuhash      : %s", bin2hex((uchar *)vhash, 32));
                        //    gpulog(LOG_DEBUG, thr_id, "data 00..09 : %s", bin2hex((uchar *)pdata, 40));
                        //    gpulog(LOG_DEBUG, thr_id, "data 10..19 : %s", bin2hex((uchar *)&pdata[10], 40));
                        //}
                    }
                    warn = 0;
                }
            }
            if (cudaHashFinalDone)
                cudaMemset(d_x16ResNonce[thr_id], 0xff, MAX_NONCES * sizeof(uint32_t));

        }

        if ((uint64_t)throughput + pdata[19] >= max_nonce) {
            pdata[19] = max_nonce;
            break;
        }

        pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

    #ifdef _PROFILE_METRICS_X16
    PRINT_METRICS
    #endif // _PROFILE_METRICS_X16

    *hashes_done = pdata[19] - first_nonce;
    return 0;
}

// cleanup
extern "C" void free_x16r(int thr_id) {
    if (!init[thr_id])
        return;

    cudaThreadSynchronize();

    cudaFree(d_hash[thr_id]);

    quark_groestl512_cpu_free(thr_id);
    x11_simd512_cpu_free(thr_id);
    //x15_whirlpool_cpu_free(thr_id);

    cuda_check_cpu_free(thr_id);

    cudaDeviceSynchronize();
    init[thr_id] = false;
}
