/*
* Shabal-512
* tpruvot 2018, based on alexis x14 and xevan kernlx code
* DumaxFr 2018
*/

#include <cuda_helper.h>
#include <cuda_vectors.h>

#include "./cuda_b_shabal512.h"

#define C32(x) (x)
#define T32(x) (x)

#define SHABAL512_TPB80 256
#define SHABAL512_TPB64 256
#define SHABAL512_TPB64F 256

#pragma region inlines

__device__ __forceinline__
void PERM_ELT(uint32_t &xa0, const uint32_t xa1,
    uint32_t &xb0, const uint32_t xb1, const uint32_t xb2, const uint32_t xb3,
    const uint32_t xc, const uint32_t xm) {

    #if __CUDA_ARCH__ >= 500
        uint32_t tmp;
        asm ("lop3.b32 %0, %1, %2, %3, 0x9A;" : "=r"(tmp) : "r"(xb2),"r"(xb3),"r"(xb1));
		xa0 = ((xa0 ^ (ROTL32(xa1, 15) * 5U) ^ xc) * 3U) ^ tmp ^ xm;
        asm ("lop3.b32 %0, %1, %2, %3, 0xC3;" : "=r"(xb0) : "r"(ROTL32(xb0, 1)),"r"(xa0),"r"(0x00));
    #else
		xa0 = T32((xa0 ^ (ROTL32(xa1, 15) * 5U) ^ xc) * 3U) ^ xb1 ^ (xb2 & ~xb3) ^ xm;
		xb0 = T32(~(ROTL32(xb0, 1) ^ xa0));
    #endif // __CUDA_ARCH__ >= 500
}

#pragma endregion

#pragma region macros

#define PERM_STEP_0_BC do { \
		PERM_ELT(A[0], A[11], B[0], B[13], B[9], B[6], C[8], M[0]); \
		PERM_ELT(A[1], A[0], B[1], B[14], B[10], B[7], C[7], M[1]); \
		PERM_ELT(A[2], A[1], B[2], B[15], B[11], B[8], C[6], M[2]); \
		PERM_ELT(A[3], A[2], B[3], B[0], B[12], B[9], C[5], M[3]); \
		PERM_ELT(A[4], A[3], B[4], B[1], B[13], B[10], C[4], M[4]); \
		PERM_ELT(A[5], A[4], B[5], B[2], B[14], B[11], C[3], M[5]); \
		PERM_ELT(A[6], A[5], B[6], B[3], B[15], B[12], C[2], M[6]); \
		PERM_ELT(A[7], A[6], B[7], B[4], B[0], B[13], C[1], M[7]); \
		PERM_ELT(A[8], A[7], B[8], B[5], B[1], B[14], C[0], M[8]); \
		PERM_ELT(A[9], A[8], B[9], B[6], B[2], B[15], C[15], M[9]); \
		PERM_ELT(A[10], A[9], B[10], B[7], B[3], B[0], C[14], M[10]); \
		PERM_ELT(A[11], A[10], B[11], B[8], B[4], B[1], C[13], M[11]); \
		PERM_ELT(A[0], A[11], B[12], B[9], B[5], B[2], C[12], M[12]); \
		PERM_ELT(A[1], A[0], B[13], B[10], B[6], B[3], C[11], M[13]); \
		PERM_ELT(A[2], A[1], B[14], B[11], B[7], B[4], C[10], M[14]); \
		PERM_ELT(A[3], A[2], B[15], B[12], B[8], B[5], C[9], M[15]); \
    } while (0)

#define PERM_STEP_0_CB do { \
		PERM_ELT(A[0], A[11], C[0], C[13], C[9], C[6], B[8], M[0]); \
		PERM_ELT(A[1], A[0], C[1], C[14], C[10], C[7], B[7], M[1]); \
		PERM_ELT(A[2], A[1], C[2], C[15], C[11], C[8], B[6], M[2]); \
		PERM_ELT(A[3], A[2], C[3], C[0], C[12], C[9], B[5], M[3]); \
		PERM_ELT(A[4], A[3], C[4], C[1], C[13], C[10], B[4], M[4]); \
		PERM_ELT(A[5], A[4], C[5], C[2], C[14], C[11], B[3], M[5]); \
		PERM_ELT(A[6], A[5], C[6], C[3], C[15], C[12], B[2], M[6]); \
		PERM_ELT(A[7], A[6], C[7], C[4], C[0], C[13], B[1], M[7]); \
		PERM_ELT(A[8], A[7], C[8], C[5], C[1], C[14], B[0], M[8]); \
		PERM_ELT(A[9], A[8], C[9], C[6], C[2], C[15], B[15], M[9]); \
		PERM_ELT(A[10], A[9], C[10], C[7], C[3], C[0], B[14], M[10]); \
		PERM_ELT(A[11], A[10], C[11], C[8], C[4], C[1], B[13], M[11]); \
		PERM_ELT(A[0], A[11], C[12], C[9], C[5], C[2], B[12], M[12]); \
		PERM_ELT(A[1], A[0], C[13], C[10], C[6], C[3], B[11], M[13]); \
		PERM_ELT(A[2], A[1], C[14], C[11], C[7], C[4], B[10], M[14]); \
		PERM_ELT(A[3], A[2], C[15], C[12], C[8], C[5], B[9], M[15]); \
    } while (0)

#define PERM_STEP_1_BC do { \
		PERM_ELT(A[4], A[3], B[0], B[13], B[9], B[6], C[8], M[0]); \
		PERM_ELT(A[5], A[4], B[1], B[14], B[10], B[7], C[7], M[1]); \
		PERM_ELT(A[6], A[5], B[2], B[15], B[11], B[8], C[6], M[2]); \
		PERM_ELT(A[7], A[6], B[3], B[0], B[12], B[9], C[5], M[3]); \
		PERM_ELT(A[8], A[7], B[4], B[1], B[13], B[10], C[4], M[4]); \
		PERM_ELT(A[9], A[8], B[5], B[2], B[14], B[11], C[3], M[5]); \
		PERM_ELT(A[10], A[9], B[6], B[3], B[15], B[12], C[2], M[6]); \
		PERM_ELT(A[11], A[10], B[7], B[4], B[0], B[13], C[1], M[7]); \
		PERM_ELT(A[0], A[11], B[8], B[5], B[1], B[14], C[0], M[8]); \
		PERM_ELT(A[1], A[0], B[9], B[6], B[2], B[15], C[15], M[9]); \
		PERM_ELT(A[2], A[1], B[10], B[7], B[3], B[0], C[14], M[10]); \
		PERM_ELT(A[3], A[2], B[11], B[8], B[4], B[1], C[13], M[11]); \
		PERM_ELT(A[4], A[3], B[12], B[9], B[5], B[2], C[12], M[12]); \
		PERM_ELT(A[5], A[4], B[13], B[10], B[6], B[3], C[11], M[13]); \
		PERM_ELT(A[6], A[5], B[14], B[11], B[7], B[4], C[10], M[14]); \
		PERM_ELT(A[7], A[6], B[15], B[12], B[8], B[5], C[9], M[15]); \
    } while (0)

#define PERM_STEP_1_CB do { \
		PERM_ELT(A[4], A[3], C[0], C[13], C[9], C[6], B[8], M[0]); \
		PERM_ELT(A[5], A[4], C[1], C[14], C[10], C[7], B[7], M[1]); \
		PERM_ELT(A[6], A[5], C[2], C[15], C[11], C[8], B[6], M[2]); \
		PERM_ELT(A[7], A[6], C[3], C[0], C[12], C[9], B[5], M[3]); \
		PERM_ELT(A[8], A[7], C[4], C[1], C[13], C[10], B[4], M[4]); \
		PERM_ELT(A[9], A[8], C[5], C[2], C[14], C[11], B[3], M[5]); \
		PERM_ELT(A[10], A[9], C[6], C[3], C[15], C[12], B[2], M[6]); \
		PERM_ELT(A[11], A[10], C[7], C[4], C[0], C[13], B[1], M[7]); \
		PERM_ELT(A[0], A[11], C[8], C[5], C[1], C[14], B[0], M[8]); \
		PERM_ELT(A[1], A[0], C[9], C[6], C[2], C[15], B[15], M[9]); \
		PERM_ELT(A[2], A[1], C[10], C[7], C[3], C[0], B[14], M[10]); \
		PERM_ELT(A[3], A[2], C[11], C[8], C[4], C[1], B[13], M[11]); \
		PERM_ELT(A[4], A[3], C[12], C[9], C[5], C[2], B[12], M[12]); \
		PERM_ELT(A[5], A[4], C[13], C[10], C[6], C[3], B[11], M[13]); \
		PERM_ELT(A[6], A[5], C[14], C[11], C[7], C[4], B[10], M[14]); \
		PERM_ELT(A[7], A[6], C[15], C[12], C[8], C[5], B[9], M[15]); \
    } while (0)

#define PERM_STEP_2_BC do { \
		PERM_ELT(A[8], A[7], B[0], B[13], B[9], B[6], C[8], M[0]); \
		PERM_ELT(A[9], A[8], B[1], B[14], B[10], B[7], C[7], M[1]); \
		PERM_ELT(A[10], A[9], B[2], B[15], B[11], B[8], C[6], M[2]); \
		PERM_ELT(A[11], A[10], B[3], B[0], B[12], B[9], C[5], M[3]); \
		PERM_ELT(A[0], A[11], B[4], B[1], B[13], B[10], C[4], M[4]); \
		PERM_ELT(A[1], A[0], B[5], B[2], B[14], B[11], C[3], M[5]); \
		PERM_ELT(A[2], A[1], B[6], B[3], B[15], B[12], C[2], M[6]); \
		PERM_ELT(A[3], A[2], B[7], B[4], B[0], B[13], C[1], M[7]); \
		PERM_ELT(A[4], A[3], B[8], B[5], B[1], B[14], C[0], M[8]); \
		PERM_ELT(A[5], A[4], B[9], B[6], B[2], B[15], C[15], M[9]); \
		PERM_ELT(A[6], A[5], B[10], B[7], B[3], B[0], C[14], M[10]); \
		PERM_ELT(A[7], A[6], B[11], B[8], B[4], B[1], C[13], M[11]); \
		PERM_ELT(A[8], A[7], B[12], B[9], B[5], B[2], C[12], M[12]); \
		PERM_ELT(A[9], A[8], B[13], B[10], B[6], B[3], C[11], M[13]); \
		PERM_ELT(A[10], A[9], B[14], B[11], B[7], B[4], C[10], M[14]); \
		PERM_ELT(A[11], A[10], B[15], B[12], B[8], B[5], C[9], M[15]); \
    } while (0)

#define PERM_STEP_2_CB do { \
		PERM_ELT(A[8], A[7], C[0], C[13], C[9], C[6], B[8], M[0]); \
		PERM_ELT(A[9], A[8], C[1], C[14], C[10], C[7], B[7], M[1]); \
		PERM_ELT(A[10], A[9], C[2], C[15], C[11], C[8], B[6], M[2]); \
		PERM_ELT(A[11], A[10], C[3], C[0], C[12], C[9], B[5], M[3]); \
		PERM_ELT(A[0], A[11], C[4], C[1], C[13], C[10], B[4], M[4]); \
		PERM_ELT(A[1], A[0], C[5], C[2], C[14], C[11], B[3], M[5]); \
		PERM_ELT(A[2], A[1], C[6], C[3], C[15], C[12], B[2], M[6]); \
		PERM_ELT(A[3], A[2], C[7], C[4], C[0], C[13], B[1], M[7]); \
		PERM_ELT(A[4], A[3], C[8], C[5], C[1], C[14], B[0], M[8]); \
		PERM_ELT(A[5], A[4], C[9], C[6], C[2], C[15], B[15], M[9]); \
		PERM_ELT(A[6], A[5], C[10], C[7], C[3], C[0], B[14], M[10]); \
		PERM_ELT(A[7], A[6], C[11], C[8], C[4], C[1], B[13], M[11]); \
		PERM_ELT(A[8], A[7], C[12], C[9], C[5], C[2], B[12], M[12]); \
		PERM_ELT(A[9], A[8], C[13], C[10], C[6], C[3], B[11], M[13]); \
		PERM_ELT(A[10], A[9], C[14], C[11], C[7], C[4], B[10], M[14]); \
		PERM_ELT(A[11], A[10], C[15], C[12], C[8], C[5], B[9], M[15]); \
    } while (0)

#define ADD_BLOCK_C do { \
		A[11] = T32(A[11] + C[6]); \
		A[10] = T32(A[10] + C[5]); \
		A[9] = T32(A[9] + C[4]); \
		A[8] = T32(A[8] + C[3]); \
		A[7] = T32(A[7] + C[2]); \
		A[6] = T32(A[6] + C[1]); \
		A[5] = T32(A[5] + C[0]); \
		A[4] = T32(A[4] + C[15]); \
		A[3] = T32(A[3] + C[14]); \
		A[2] = T32(A[2] + C[13]); \
		A[1] = T32(A[1] + C[12]); \
		A[0] = T32(A[0] + C[11]); \
		A[11] = T32(A[11] + C[10]); \
		A[10] = T32(A[10] + C[9]); \
		A[9] = T32(A[9] + C[8]); \
		A[8] = T32(A[8] + C[7]); \
		A[7] = T32(A[7] + C[6]); \
		A[6] = T32(A[6] + C[5]); \
		A[5] = T32(A[5] + C[4]); \
		A[4] = T32(A[4] + C[3]); \
		A[3] = T32(A[3] + C[2]); \
		A[2] = T32(A[2] + C[1]); \
		A[1] = T32(A[1] + C[0]); \
		A[0] = T32(A[0] + C[15]); \
		A[11] = T32(A[11] + C[14]); \
		A[10] = T32(A[10] + C[13]); \
		A[9] = T32(A[9] + C[12]); \
		A[8] = T32(A[8] + C[11]); \
		A[7] = T32(A[7] + C[10]); \
		A[6] = T32(A[6] + C[9]); \
		A[5] = T32(A[5] + C[8]); \
		A[4] = T32(A[4] + C[7]); \
		A[3] = T32(A[3] + C[6]); \
		A[2] = T32(A[2] + C[5]); \
		A[1] = T32(A[1] + C[4]); \
		A[0] = T32(A[0] + C[3]); \
    } while (0)

#define ADD_BLOCK_B do { \
		A[11] = T32(A[11] + B[6]); \
		A[10] = T32(A[10] + B[5]); \
		A[9] = T32(A[9] + B[4]); \
		A[8] = T32(A[8] + B[3]); \
		A[7] = T32(A[7] + B[2]); \
		A[6] = T32(A[6] + B[1]); \
		A[5] = T32(A[5] + B[0]); \
		A[4] = T32(A[4] + B[15]); \
		A[3] = T32(A[3] + B[14]); \
		A[2] = T32(A[2] + B[13]); \
		A[1] = T32(A[1] + B[12]); \
		A[0] = T32(A[0] + B[11]); \
		A[11] = T32(A[11] + B[10]); \
		A[10] = T32(A[10] + B[9]); \
		A[9] = T32(A[9] + B[8]); \
		A[8] = T32(A[8] + B[7]); \
		A[7] = T32(A[7] + B[6]); \
		A[6] = T32(A[6] + B[5]); \
		A[5] = T32(A[5] + B[4]); \
		A[4] = T32(A[4] + B[3]); \
		A[3] = T32(A[3] + B[2]); \
		A[2] = T32(A[2] + B[1]); \
		A[1] = T32(A[1] + B[0]); \
		A[0] = T32(A[0] + B[15]); \
		A[11] = T32(A[11] + B[14]); \
		A[10] = T32(A[10] + B[13]); \
		A[9] = T32(A[9] + B[12]); \
		A[8] = T32(A[8] + B[11]); \
		A[7] = T32(A[7] + B[10]); \
		A[6] = T32(A[6] + B[9]); \
		A[5] = T32(A[5] + B[8]); \
		A[4] = T32(A[4] + B[7]); \
		A[3] = T32(A[3] + B[6]); \
		A[2] = T32(A[2] + B[5]); \
		A[1] = T32(A[1] + B[4]); \
		A[0] = T32(A[0] + B[3]); \
    } while (0)

#define APPLY_P_BC do { \
		PERM_STEP_0_BC; \
		PERM_STEP_1_BC; \
		PERM_STEP_2_BC; \
        ADD_BLOCK_C; \
	} while (0)

#define APPLY_P_CB do { \
		PERM_STEP_0_CB; \
		PERM_STEP_1_CB; \
		PERM_STEP_2_CB; \
        ADD_BLOCK_B; \
	} while (0)

#pragma endregion

#pragma region SHAbal512_80

__constant__ static uint32_t c_message80[20];

__host__
void cuda_base_shabal512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_message80, pdata, sizeof(c_message80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(SHABAL512_TPB80, 4)
void shabal512_gpu_hash_80(uint32_t threads, const uint32_t startNonce, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t A[] = {
		0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632,
        0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
		0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F
    };
	uint32_t B[] = {
		0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640,
        0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
		0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E,
        0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B
    };
	uint32_t C[] = {
		0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359,
        0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
		0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A,
        0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969
    };

	if (thread < threads) {

        uint32_t M[16];

        #pragma unroll
        for (int idx = 0; idx < 16; idx += 4)
            AS_UINT4(&M[idx]) = AS_UINT4(&c_message80[idx]);

		//INPUT_BLOCK_ADD;
		*(uint16*)&B[0] += *(uint16*)&M[0];
        A[0] ^= 1;
        #pragma unroll
	    for (int i = 0; i < 16; i++) {
		    B[i] = ROTL32(B[i], 17);
	    }
		APPLY_P_BC;
		//INPUT_BLOCK_SUB;
		*(uint16*)&C[0] -= *(uint16*)&M[0];

		M[0] = c_message80[16];
		M[1] = c_message80[17];
		M[2] = c_message80[18];
		M[3] = cuda_swab32(startNonce + thread);
		M[4] = 0x80;
		M[5] = M[6] = M[7] = M[8] = M[9] = M[10] = M[11] = M[12] = M[13] = M[14] = M[15] = 0;

		//INPUT_BLOCK_ADD;
		*(uint4*)&C[0] += *(uint4*)&M[0];
        C[4] += M[4];
        A[0] ^= 2;
        #pragma unroll
	    for (int i = 0; i < 16; i++) {
		    C[i] = ROTL32(C[i], 17);
	    }
		APPLY_P_CB;

        A[0] ^= 2;
        #pragma unroll
	    for (int i = 0; i < 16; i++) {
		    B[i] = ROTL32(B[i], 17);
	    }
		APPLY_P_BC;

        A[0] ^= 2;
        #pragma unroll
	    for (int i = 0; i < 16; i++) {
		    C[i] = ROTL32(C[i], 17);
	    }
		APPLY_P_CB;

        A[0] ^= 2;
        #pragma unroll
	    for (int i = 0; i < 16; i++) {
		    B[i] = ROTL32(B[i], 17);
	    }
        PERM_STEP_0_BC;
        PERM_STEP_1_BC;
        PERM_STEP_2_BC;

		uint32_t *outHash = &g_hash[thread<<4];
        #pragma unroll
        for (int i = 0; i < 16; i += 4)
            AS_UINT4(&outHash[i]) = AS_UINT4(&B[i]);
	}
}

__host__
void cuda_base_shabal512_cpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = SHABAL512_TPB80;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	shabal512_gpu_hash_80<<<grid, block >>>(threads, startNonce, d_hash);
}

#pragma endregion

#pragma region SHAbal512_64

__global__ __launch_bounds__(SHABAL512_TPB64, 4)
void shabal512_gpu_hash_64(uint32_t threads, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t A[] = {
		0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632,
        0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
		0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F
    };
	uint32_t B[] = {
		0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640,
        0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
		0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E,
        0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B
    };
	uint32_t C[] = {
		0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359,
        0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
		0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A,
        0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969
    };

	if (thread < threads) {

        uint32_t M[16];

        uint32_t *pHash = &g_hash[thread<<4];
        #pragma unroll 4
        for (int idx = 0; idx < 16; idx += 4)
            AS_UINT4(&M[idx]) = AS_UINT4(&pHash[idx]);

		*(uint16*)&B[0] += *(uint16*)&M[0];
        A[0] ^= 1;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            B[i] = ROTL32(B[i], 17);
        }
        APPLY_P_BC;
		*(uint16*)&C[0] -= *(uint16*)&M[0];

        M[0] = 0x80;
        M[1] = M[2] = M[3] = M[4] = M[5] = M[6] = M[7] = M[8] = M[9] = M[10] = M[11] = M[12] = M[13] = M[14] = M[15] = 0;

        C[0] += M[0];
        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            C[i] = ROTL32(C[i], 17);
        }
        APPLY_P_CB;

        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            B[i] = ROTL32(B[i], 17);
        }
        APPLY_P_BC;

        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            C[i] = ROTL32(C[i], 17);
        }
        APPLY_P_CB;

        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            B[i] = ROTL32(B[i], 17);
        }
        PERM_STEP_0_BC;
        PERM_STEP_1_BC;
        PERM_STEP_2_BC;

		uint32_t *outHash = pHash;
        #pragma unroll 4
        for (int i = 0; i < 16; i += 4)
            AS_UINT4(&outHash[i]) = AS_UINT4(&B[i]);
	}
}

__host__
void cuda_base_shabal512_cpu_hash_64(const uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = SHABAL512_TPB64;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	shabal512_gpu_hash_64<<<grid, block >>>(threads, d_hash);
}

#pragma endregion

#pragma region SHAbal512_64_final

__global__ __launch_bounds__(SHABAL512_TPB64F, 4)
void shabal512_gpu_hash_64_final(const uint32_t threads, const uint32_t* __restrict__ g_hash, const uint32_t startNonce, uint32_t *resNonce, const uint64_t target) {

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t A[] = {
		0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632,
        0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
		0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F
    };
	uint32_t B[] = {
		0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640,
        0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
		0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E,
        0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B
    };
	uint32_t C[] = {
		0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359,
        0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
		0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A,
        0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969
    };

	if (thread < threads) {

        uint32_t M[16];

        const uint32_t* __restrict__ pHash = &g_hash[thread<<4];
        #pragma unroll 4
        for (int idx = 0; idx < 16; idx += 4)
            AS_UINT4(&M[idx]) = AS_UINT4(&pHash[idx]);

		*(uint16*)&B[0] += *(uint16*)&M[0];
        A[0] ^= 1;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            B[i] = ROTL32(B[i], 17);
        }
        APPLY_P_BC;
		*(uint16*)&C[0] -= *(uint16*)&M[0];

        M[0] = 0x80;
        M[1] = M[2] = M[3] = M[4] = M[5] = M[6] = M[7] = M[8] = M[9] = M[10] = M[11] = M[12] = M[13] = M[14] = M[15] = 0;

        C[0] += M[0];
        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            C[i] = ROTL32(C[i], 17);
        }
        APPLY_P_CB;

        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            B[i] = ROTL32(B[i], 17);
        }
        APPLY_P_BC;

        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            C[i] = ROTL32(C[i], 17);
        }
        APPLY_P_CB;

        A[0] ^= 2;
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            B[i] = ROTL32(B[i], 17);
        }
        PERM_STEP_0_BC;
        PERM_STEP_1_BC;
		PERM_ELT(A[8], A[7], B[0], B[13], B[9], B[6], C[8], M[0]);
		PERM_ELT(A[9], A[8], B[1], B[14], B[10], B[7], C[7], M[1]);
		PERM_ELT(A[10], A[9], B[2], B[15], B[11], B[8], C[6], M[2]);
		PERM_ELT(A[11], A[10], B[3], B[0], B[12], B[9], C[5], M[3]);
		PERM_ELT(A[0], A[11], B[4], B[1], B[13], B[10], C[4], M[4]);
		PERM_ELT(A[1], A[0], B[5], B[2], B[14], B[11], C[3], M[5]);
        PERM_ELT(A[2], A[1], B[6], B[3], B[15], B[12], C[2], M[6]);
		PERM_ELT(A[3], A[2], B[7], B[4], B[0], B[13], C[1], M[7]);

        uint64_t check = *(uint64_t*)&B[6];
		if (check <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], startNonce + thread);
            if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

__host__
void cuda_base_shabal512_cpu_hash_64f(const uint32_t threads, uint32_t *d_hash, const uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target) {

	const uint32_t threadsperblock = SHABAL512_TPB64F;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	shabal512_gpu_hash_64_final<<<grid, block >>>(threads, d_hash, startNonce, d_resNonce, target);
}

#pragma endregion
