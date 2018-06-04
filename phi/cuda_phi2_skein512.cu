/* SKEIN 64 and 80 based on Alexis Provos version */

#include <stdio.h>
#include <cuda_vectors.h>
#include <cuda_vector_uint2x4.h>

#include "./cuda_phi2_skein512.h"


#define PHI2SKEIN512_TPB64  512
#define PHI2SKEIN512_TPB64F 512


#pragma region macros

/*
 * M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
 */

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
 * M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
 */

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define SKBI(k, s, i)   XCAT(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + make_uint2(s,0); \
}

#define TFBIG_MIX(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROL2(x1, rc) ^ x0; \
}

#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX(w0, w1, rc0); \
	TFBIG_MIX(w2, w3, rc1); \
	TFBIG_MIX(w4, w5, rc2); \
	TFBIG_MIX(w6, w7, rc3); \
}

#define TFBIG_4e(s)  { \
	TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4o(s)  { \
	TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

/* uint2 variant for SM3.2+ */

#define TFBIG_KINIT_UI2(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
	k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
		^ vectorize(0x1BD11BDAA9FC1A22); \
	t2 = t0 ^ t1; \
}

#define TFBIG_ADDKEY_UI2(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + vectorize(s)); \
}

#define TFBIG_ADDKEY_PRE(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + (s)); \
}

#define TFBIG_MIX_UI2(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROL2(x1, rc) ^ x0; \
}

#define TFBIG_MIX_PRE(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROTL64(x1, rc) ^ x0; \
}

#define TFBIG_MIX8_UI2(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX_UI2(w0, w1, rc0); \
	TFBIG_MIX_UI2(w2, w3, rc1); \
	TFBIG_MIX_UI2(w4, w5, rc2); \
	TFBIG_MIX_UI2(w6, w7, rc3); \
}

#define TFBIG_MIX8_PRE(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX_PRE(w0, w1, rc0); \
	TFBIG_MIX_PRE(w2, w3, rc1); \
	TFBIG_MIX_PRE(w4, w5, rc2); \
	TFBIG_MIX_PRE(w6, w7, rc3); \
}

#define TFBIG_4e_UI2(s) { \
	TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4e_PRE(s) { \
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4o_UI2(s) { \
	TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

#define TFBIG_4o_PRE(s) { \
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

#define macro1() {\
	p[0] += p[1]; p[2] += p[3]; p[4] += p[5]; p[6] += p[7]; p[1] = ROL2(p[1],46) ^ p[0]; \
	p[3] = ROL2(p[3],36) ^ p[2]; p[5] = ROL2(p[5],19) ^ p[4]; p[7] = ROL2(p[7], 37) ^ p[6]; \
	p[2] += p[1]; p[4] += p[7]; p[6] += p[5]; p[0] += p[3]; p[1] = ROL2(p[1],33) ^ p[2]; \
	p[7] = ROL2(p[7],27) ^ p[4]; p[5] = ROL2(p[5],14) ^ p[6]; p[3] = ROL2(p[3], 42) ^ p[0]; \
	p[4] += p[1]; p[6] += p[3]; p[0] += p[5]; p[2] += p[7]; p[1] = ROL2(p[1],17) ^ p[4]; \
	p[3] = ROL2(p[3],49) ^ p[6]; p[5] = ROL2(p[5],36) ^ p[0]; p[7] = ROL2(p[7], 39) ^ p[2]; \
	p[6] += p[1]; p[0] += p[7]; p[2] += p[5]; p[4] += p[3]; p[1] = ROL2(p[1],44) ^ p[6]; \
	p[7] = ROL2(p[7], 9) ^ p[0]; p[5] = ROL2(p[5],54) ^ p[2]; p[3] = ROR8(p[3]) ^ p[4]; \
}

#define macro2() { \
	p[0] += p[1]; p[2] += p[3]; p[4] += p[5]; p[6] += p[7]; p[1] = ROL2(p[1], 39) ^ p[0]; \
	p[3] = ROL2(p[3], 30) ^ p[2]; p[5] = ROL2(p[5], 34) ^ p[4]; p[7] = ROL24(p[7]) ^ p[6]; \
	p[2] += p[1]; p[4] += p[7]; p[6] += p[5]; p[0] += p[3]; p[1] = ROL2(p[1], 13) ^ p[2]; \
	p[7] = ROL2(p[7], 50) ^ p[4]; p[5] = ROL2(p[5], 10) ^ p[6]; p[3] = ROL2(p[3], 17) ^ p[0]; \
	p[4] += p[1]; p[6] += p[3]; p[0] += p[5]; p[2] += p[7]; p[1] = ROL2(p[1], 25) ^ p[4]; \
	p[3] = ROL2(p[3], 29) ^ p[6]; p[5] = ROL2(p[5], 39) ^ p[0]; p[7] = ROL2(p[7], 43) ^ p[2]; \
	p[6] += p[1]; p[0] += p[7]; p[2] += p[5]; p[4] += p[3]; p[1] = ROL8(p[1]) ^ p[6]; \
	p[7] = ROL2(p[7], 35) ^ p[0]; p[5] = ROR8(p[5]) ^ p[2]; p[3] = ROL2(p[3], 22) ^ p[4]; \
}

#define macro3() { \
	hash64[0]+= hash64[1]; hash64[2]+= hash64[3]; hash64[4]+= hash64[5]; hash64[6]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 39) ^ hash64[0]; \
	hash64[3] = ROL2(hash64[3], 30) ^ hash64[2]; \
	hash64[5] = ROL2(hash64[5], 34) ^ hash64[4]; \
	hash64[7] = ROL24(hash64[7]) ^ hash64[6]; \
	hash64[2]+= hash64[1]; hash64[4]+= hash64[7]; hash64[6]+= hash64[5]; hash64[0]+= hash64[3]; \
	hash64[1] = ROL2(hash64[1], 13) ^ hash64[2]; \
	hash64[7] = ROL2(hash64[7], 50) ^ hash64[4]; \
	hash64[5] = ROL2(hash64[5], 10) ^ hash64[6]; \
	hash64[3] = ROL2(hash64[3], 17) ^ hash64[0]; \
	hash64[4]+= hash64[1]; hash64[6]+= hash64[3]; hash64[0]+= hash64[5]; hash64[2]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 25) ^ hash64[4]; \
	hash64[3] = ROL2(hash64[3], 29) ^ hash64[6]; \
	hash64[5] = ROL2(hash64[5], 39) ^ hash64[0]; \
	hash64[7] = ROL2(hash64[7], 43) ^ hash64[2]; \
	hash64[6]+= hash64[1]; hash64[0]+= hash64[7]; hash64[2]+= hash64[5]; hash64[4]+= hash64[3]; \
	hash64[1] = ROL8(hash64[1]) ^ hash64[6]; \
	hash64[7] = ROL2(hash64[7], 35) ^ hash64[0]; \
	hash64[5] = ROR8(hash64[5]) ^ hash64[2]; \
	hash64[3] = ROL2(hash64[3], 22) ^ hash64[4]; \
}

#define macro4() {\
	hash64[0]+= hash64[1]; hash64[2]+= hash64[3]; hash64[4]+= hash64[5]; hash64[6]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 46) ^ hash64[0]; \
	hash64[3] = ROL2(hash64[3], 36) ^ hash64[2]; \
	hash64[5] = ROL2(hash64[5], 19) ^ hash64[4]; \
	hash64[7] = ROL2(hash64[7], 37) ^ hash64[6]; \
	hash64[2]+= hash64[1]; hash64[4]+= hash64[7]; hash64[6]+= hash64[5]; hash64[0]+= hash64[3]; \
	hash64[1] = ROL2(hash64[1], 33) ^ hash64[2]; \
	hash64[7] = ROL2(hash64[7], 27) ^ hash64[4]; \
	hash64[5] = ROL2(hash64[5], 14) ^ hash64[6]; \
	hash64[3] = ROL2(hash64[3], 42) ^ hash64[0]; \
	hash64[4]+= hash64[1]; hash64[6]+= hash64[3]; hash64[0]+= hash64[5]; hash64[2]+= hash64[7]; \
	hash64[1] = ROL2(hash64[1], 17) ^ hash64[4]; \
	hash64[3] = ROL2(hash64[3], 49) ^ hash64[6]; \
	hash64[5] = ROL2(hash64[5], 36) ^ hash64[0]; \
	hash64[7] = ROL2(hash64[7], 39) ^ hash64[2]; \
	hash64[6]+= hash64[1]; hash64[0]+= hash64[7]; hash64[2]+= hash64[5]; hash64[4]+= hash64[3]; \
	hash64[1] = ROL2(hash64[1], 44) ^ hash64[6]; \
	hash64[7] = ROL2(hash64[7], 9) ^ hash64[0]; \
	hash64[5] = ROL2(hash64[5], 54) ^ hash64[2]; \
	hash64[3] = ROR8(hash64[3]) ^ hash64[4]; \
}

#pragma endregion

static __constant__ const uint2 buffer[112] = {
	{0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C434, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC33, 0xAE18A40B},
	{0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599},
	{0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173EC5, 0xCAB2076D},
	{0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C},
	{0x1A75B523, 0x991112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51D0, 0x4903ADFF},
	{0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7},
	{0x660FCC33, 0xAE18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF06, 0x0D95DE39},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B},
	{0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF43, 0xFD95DE39}, {0x27C79BD2, 0x8FD19341},
	{0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B}, {0x98173EC4, 0xCAB2076D},
	{0x749C51CE, 0x4903ADFF}, {0x9746DF43, 0xFD95DE39}, {0x27C79C0E, 0x8FD19341}, {0xFF352CB6, 0x9A255629},
	{0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B}, {0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x4903ADFF},
	{0x9746DF03, 0x0D95DE39}, {0x27C79C0E, 0x8FD19341}, {0xFF352CB1, 0x8A255629}, {0xDF6CA7B6, 0x5DB62599},
	{0x660FCC33, 0xAE18A40B}, {0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39},
	{0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x8A255629}, {0xDF6CA7F0, 0x4DB62599}, {0xA9D5C3FB, 0xEABE394C},
	{0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341},
	{0xFF352CB1, 0x9A255629}, {0xDF6CA7F0, 0x4DB62599}, {0xA9D5C434, 0xEABE394C}, {0x1A75B52B, 0x991112C7},
	{0x749C51CE, 0x4903ADFF}, {0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C434, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC3C, 0xAE18A40B},
	{0x9746DF03, 0x0D95DE39}, {0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599},
	{0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x891112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173ece, 0xcab2076d},
	{0x27C79BCE, 0x8FD19341}, {0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C},
	{0x1A75B523, 0x991112C7}, {0x660FCC73, 0x9E18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51D9, 0x4903ADFF},
	{0xFF352CB1, 0x9A255629}, {0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7},
	{0x660FCC33, 0xAE18A40B}, {0x98173F04, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF0F, 0x0D95DE39},
	{0xDF6CA7B0, 0x5DB62599}, {0xA9D5C3F4, 0xEABE394C}, {0x1A75B523, 0x991112C7}, {0x660FCC33, 0xAE18A40B},
	{0x98173EC4, 0xCAB2076D}, {0x749C51CE, 0x3903ADFF}, {0x9746DF43, 0xFD95DE39}, {0x27C79BDB, 0x8FD19341}
};

#pragma region Phi2_Skein512_64

__global__
__launch_bounds__(PHI2SKEIN512_TPB64, 3)
void cuda_phi2_skein512_gpu_hash_64(const uint32_t threads, uint32_t *g_hash) {

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads) {

		// Skein
		uint2 p[8], h[9];

		uint32_t *Hash = &g_hash[thread<<4];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&p[0] = __ldg4(&phash[0]);
		*(uint2x4*)&p[4] = __ldg4(&phash[1]);

		h[0] = p[0];	h[1] = p[1];	h[2] = p[2];	h[3] = p[3];
		h[4] = p[4];	h[5] = p[5];	h[6] = p[6];	h[7] = p[7];

		p[0] += buffer[  0];	p[1] += buffer[  1];	p[2] += buffer[  2];	p[3] += buffer[3];
		p[4] += buffer[  4];	p[5] += buffer[  5];	p[6] += buffer[  6];	p[7] += buffer[7];
		macro1();
		p[0] += buffer[  8];	p[1] += buffer[  9];	p[2] += buffer[ 10];	p[3] += buffer[11];
		p[4] += buffer[ 12];	p[5] += buffer[ 13];	p[6] += buffer[ 14];	p[7] += buffer[15];
		macro2();
		p[0] += buffer[ 16];	p[1] += buffer[ 17];	p[2] += buffer[ 18];	p[3] += buffer[19];
		p[4] += buffer[ 20];	p[5] += buffer[ 21];	p[6] += buffer[ 22];	p[7] += buffer[23];
		macro1();
		p[0] += buffer[ 24];	p[1] += buffer[ 25];	p[2] += buffer[ 26];	p[3] += buffer[27];
		p[4] += buffer[ 28];	p[5] += buffer[ 29];	p[6] += buffer[ 30];	p[7] += buffer[31];
		macro2();
		p[0] += buffer[ 32];	p[1] += buffer[ 33];	p[2] += buffer[ 34];	p[3] += buffer[35];
		p[4] += buffer[ 36];	p[5] += buffer[ 37];	p[6] += buffer[ 38];	p[7] += buffer[39];
		macro1();
		p[0] += buffer[ 40];	p[1] += buffer[ 41];	p[2] += buffer[ 42];	p[3] += buffer[43];
		p[4] += buffer[ 44];	p[5] += buffer[ 45];	p[6] += buffer[ 46];	p[7] += buffer[47];
		macro2();
		p[0] += buffer[ 48];	p[1] += buffer[ 49];	p[2] += buffer[ 50];	p[3] += buffer[51];
		p[4] += buffer[ 52];	p[5] += buffer[ 53];	p[6] += buffer[ 54];	p[7] += buffer[55];
		macro1();
		p[0] += buffer[ 56];	p[1] += buffer[ 57];	p[2] += buffer[ 58];	p[3] += buffer[59];
		p[4] += buffer[ 60];	p[5] += buffer[ 61];	p[6] += buffer[ 62];	p[7] += buffer[63];
		macro2();
		p[0] += buffer[ 64];	p[1] += buffer[ 65];	p[2] += buffer[ 66];	p[3] += buffer[67];
		p[4] += buffer[ 68];	p[5] += buffer[ 69];	p[6] += buffer[ 70];	p[7] += buffer[71];
		macro1();
		p[0] += buffer[ 72];	p[1] += buffer[ 73];	p[2] += buffer[ 74];	p[3] += buffer[75];
		p[4] += buffer[ 76];	p[5] += buffer[ 77];	p[6] += buffer[ 78];	p[7] += buffer[79];
		macro2();
		p[0] += buffer[ 80];	p[1] += buffer[ 81];	p[2] += buffer[ 82];	p[3] += buffer[83];
		p[4] += buffer[ 84];	p[5] += buffer[ 85];	p[6] += buffer[ 86];	p[7] += buffer[87];
		macro1();
		p[0] += buffer[ 88];	p[1] += buffer[ 89];	p[2] += buffer[ 90];	p[3] += buffer[91];
		p[4] += buffer[ 92];	p[5] += buffer[ 93];	p[6] += buffer[ 94];	p[7] += buffer[95];
		macro2();
		p[0] += buffer[ 96];	p[1] += buffer[ 97];	p[2] += buffer[ 98];	p[3] += buffer[99];
		p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		macro1();
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];
		p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		macro2();
		p[0]+= make_uint2(0xA9D5C3F4, 0xEABE394C);	p[1]+= make_uint2(0x1A75B523, 0x991112C7);
		p[2]+= make_uint2(0x660FCC33, 0xAE18A40B);	p[3]+= make_uint2(0x98173EC4, 0xCAB2076D);
		p[4]+= make_uint2(0x749C51CE, 0x4903ADFF);	p[5]+= make_uint2(0x9746DF43, 0xFD95DE39);
		p[6]+= make_uint2(0x27C79C0E, 0x8FD19341);	p[7]+= make_uint2(0xFF352CBF, 0x9A255629);
		macro1();
		p[0]+= make_uint2(0x1A75B523, 0x991112C7);	p[1]+= make_uint2(0x660FCC33, 0xAE18A40B);
		p[2]+= make_uint2(0x98173EC4, 0xCAB2076D);	p[3]+= make_uint2(0x749C51CE, 0x4903ADFF);
		p[4]+= make_uint2(0x9746DF03, 0x0D95DE39);	p[5]+= make_uint2(0x27C79C0E, 0x8FD19341);
		p[6]+= make_uint2(0xFF352CB1, 0x8A255629);	p[7]+= make_uint2(0xDF6CA7BF, 0x5DB62599);
		macro2();
		p[0] += vectorize(0xAE18A40B660FCC33);		p[1] += vectorize(0xcab2076d98173ec4);
		p[2] += vectorize(0x4903ADFF749C51CE);		p[3] += vectorize(0x0D95DE399746DF03);
		p[4] += vectorize(0x8FD1934127C79BCE);		p[5] += vectorize(0x8A255629FF352CB1);
		p[6] += vectorize(0x4DB62599DF6CA7F0);		p[7] += vectorize(0xEABE394CA9D5C3F4 + 16);
		macro1();
		p[0] += vectorize(0xcab2076d98173ec4);		p[1] += vectorize(0x4903ADFF749C51CE);
		p[2] += vectorize(0x0D95DE399746DF03);		p[3] += vectorize(0x8FD1934127C79BCE);
		p[4] += vectorize(0x9A255629FF352CB1);		p[5] += vectorize(0x4DB62599DF6CA7F0);
		p[6] += vectorize(0xEABE394CA9D5C3F4 + 0x0000000000000040);
		p[7] += vectorize(0x991112C71A75B523 + 17);
		macro2();
		p[0] += vectorize(0x4903ADFF749C51CE);		p[1] += vectorize(0x0D95DE399746DF03);
		p[2] += vectorize(0x8FD1934127C79BCE);		p[3] += vectorize(0x9A255629FF352CB1);
		p[4] += vectorize(0x5DB62599DF6CA7B0);		p[5] += vectorize(0xEABE394CA9D5C3F4 + 0x0000000000000040);
		p[6] += vectorize(0x891112C71A75B523);		p[7] += vectorize(0xAE18A40B660FCC33 + 18);

		#define h0 p[0]
		#define h1 p[1]
		#define h2 p[2]
		#define h3 p[3]
		#define h4 p[4]
		#define h5 p[5]
		#define h6 p[6]
		#define h7 p[7]

		h0 ^= h[0];	h1 ^= h[1];	h2 ^= h[2];	h3 ^= h[3];
		h4 ^= h[4];	h5 ^= h[5];	h6 ^= h[6];	h7 ^= h[7];

		uint2 skein_h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ vectorize(0x1BD11BDAA9FC1A22);

		uint2 hash64[8];

		hash64[5] = h5 + 8u;

		hash64[0] = h0 + h1;
		hash64[1] = ROL2(h1, 46) ^ hash64[0];
		hash64[2] = h2 + h3;
		hash64[3] = ROL2(h3, 36) ^ hash64[2];
		hash64[4] = h4 + hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] = (h6 + h7 + make_uint2(0,0xff000000));
		hash64[7] = ROL2(h7, 37) ^ hash64[6];
		hash64[2]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4]+= hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];

		hash64[0]+= h1; hash64[1]+= h2; hash64[2]+= h3; hash64[3]+= h4;
		hash64[4]+= h5;
		hash64[5]+= h6 + make_uint2(0,0xff000000);
		hash64[6]+= h7 + vectorize(0xff00000000000008);
		hash64[7]+= skein_h8 + 1u;
		macro3();
		hash64[0]+= h2; hash64[1]+= h3; hash64[2]+= h4; hash64[3]+= h5;
		hash64[4]+= h6;
		hash64[5]+= h7 + vectorize(0xff00000000000008);
		hash64[6]+= skein_h8 + 8u;
		hash64[7]+= h0 + 2u;
		macro4();
		hash64[0] = (hash64[0] + h3);			hash64[1] = (hash64[1] + h4);
		hash64[2] = (hash64[2] + h5);			hash64[3] = (hash64[3] + h6);
		hash64[4] = (hash64[4] + h7);			hash64[5] = (hash64[5] + skein_h8 + 8u);
		hash64[6] = (hash64[6] + h0 + make_uint2(0,0xff000000));
		hash64[7] = (hash64[7] + h1 + 3u);
		macro3();
		hash64[0] = (hash64[0] + h4);			hash64[1] = (hash64[1] + h5);
		hash64[2] = (hash64[2] + h6);			hash64[3] = (hash64[3] + h7);
		hash64[4] = (hash64[4] + skein_h8);		hash64[5] = (hash64[5] + h0 + make_uint2(0,0xff000000));
		hash64[6] = (hash64[6] + h1 + vectorize(0xff00000000000008));
		hash64[7] = (hash64[7] + h2 + 4u);
		macro4();
		hash64[0] = (hash64[0] + h5);			hash64[1] = (hash64[1] + h6);
		hash64[2] = (hash64[2] + h7);			hash64[3] = (hash64[3] + skein_h8);
		hash64[4] = (hash64[4] + h0);			hash64[5] = (hash64[5] + h1 + vectorize(0xff00000000000008));
		hash64[6] = (hash64[6] + h2 + 8u);		hash64[7] = (hash64[7] + h3 + 5u);
		macro3();
		hash64[0] = (hash64[0] + h6);			hash64[1] = (hash64[1] + h7);
		hash64[2] = (hash64[2] + skein_h8);		hash64[3] = (hash64[3] + h0);
		hash64[4] = (hash64[4] + h1);			hash64[5] = (hash64[5] + h2 + 8u);
		hash64[6] = (hash64[6] + h3 + make_uint2(0,0xff000000));
		hash64[7] = (hash64[7] + h4 + 6u);
		macro4();
		hash64[0] = (hash64[0] + h7);			hash64[1] = (hash64[1] + skein_h8);
		hash64[2] = (hash64[2] + h0);			hash64[3] = (hash64[3] + h1);
		hash64[4] = (hash64[4] + h2);			hash64[5] = (hash64[5] + h3 + make_uint2(0,0xff000000));
		hash64[6] = (hash64[6] + h4 + vectorize(0xff00000000000008));
		hash64[7] = (hash64[7] + h5 + 7u);
		macro3();
		hash64[0] = (hash64[0] + skein_h8);		hash64[1] = (hash64[1] + h0);
		hash64[2] = (hash64[2] + h1);			hash64[3] = (hash64[3] + h2);
		hash64[4] = (hash64[4] + h3);			hash64[5] = (hash64[5] + h4 + vectorize(0xff00000000000008));
		hash64[6] = (hash64[6] + h5 + 8u);		hash64[7] = (hash64[7] + h6 + 8u);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h0));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h1));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h2));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h3));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h4));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h5) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h6) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h7) + 9);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h1));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h2));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h3));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h4));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h5));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h6) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h7) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(skein_h8) + 10);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h2));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h3));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h4));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h5));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h6));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h7) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(skein_h8) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h0) + 11);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h3));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h4));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h5));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h6));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h7));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(skein_h8) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h0) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h1) + 12);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h4));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h5));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h6));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h7));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(skein_h8));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h0) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h1) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h2) + 13);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h5));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h6));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h7));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(skein_h8));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h0));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h1) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h2) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h3) + 14);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h6));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h7));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(skein_h8));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h0));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h1));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h2) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h3) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h4) + 15);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h7));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(skein_h8));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h0));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h1));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h2));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h3) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h4) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h5) + 16);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(skein_h8));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h0));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h1));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h2));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h3));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h4) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h5) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h6) + 17);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h0));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h1));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h2));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h3));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h4));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h5) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h6) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h7) + 18);

        // Phi2 special xor
        for (int i = 0; i < 4; i++) {
            hash64[i] ^= hash64[i+4];
        }

		uint2x4 *outpt = (uint2x4*)Hash;
		outpt[0] = *(uint2x4*)&hash64[0];
		outpt[1] = *(uint2x4*)&hash64[4];

		#undef h0
		#undef h1
		#undef h2
		#undef h3
		#undef h4
		#undef h5
		#undef h6
		#undef h7
	}
}

__host__
void cuda_phi2_skein512_cpu_hash_64(const uint32_t threads, uint32_t *d_hash) {

	uint32_t tpb = PHI2SKEIN512_TPB64;

    const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	cuda_phi2_skein512_gpu_hash_64<<<grid, block >>>(threads, d_hash);

}


#pragma endregion

#pragma region Phi2_Skein512_64_final

__global__
__launch_bounds__(PHI2SKEIN512_TPB64F, 2)
void cuda_phi2_skein512_gpu_hash_64_final(const uint32_t threads, const uint32_t* __restrict__ g_hash, const uint32_t startNonce, uint32_t *resNonce, const uint64_t target) {

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads) {

		// Skein
		uint2 p[8], h[9];

		const uint32_t *Hash = &g_hash[thread<<4];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&p[0] = __ldg4(&phash[0]);
		*(uint2x4*)&p[4] = __ldg4(&phash[1]);

		h[0] = p[0];	h[1] = p[1];	h[2] = p[2];	h[3] = p[3];
		h[4] = p[4];	h[5] = p[5];	h[6] = p[6];	h[7] = p[7];

		p[0] += buffer[  0];	p[1] += buffer[  1];	p[2] += buffer[  2];	p[3] += buffer[3];
		p[4] += buffer[  4];	p[5] += buffer[  5];	p[6] += buffer[  6];	p[7] += buffer[7];
		macro1();
		p[0] += buffer[  8];	p[1] += buffer[  9];	p[2] += buffer[ 10];	p[3] += buffer[11];
		p[4] += buffer[ 12];	p[5] += buffer[ 13];	p[6] += buffer[ 14];	p[7] += buffer[15];
		macro2();
		p[0] += buffer[ 16];	p[1] += buffer[ 17];	p[2] += buffer[ 18];	p[3] += buffer[19];
		p[4] += buffer[ 20];	p[5] += buffer[ 21];	p[6] += buffer[ 22];	p[7] += buffer[23];
		macro1();
		p[0] += buffer[ 24];	p[1] += buffer[ 25];	p[2] += buffer[ 26];	p[3] += buffer[27];
		p[4] += buffer[ 28];	p[5] += buffer[ 29];	p[6] += buffer[ 30];	p[7] += buffer[31];
		macro2();
		p[0] += buffer[ 32];	p[1] += buffer[ 33];	p[2] += buffer[ 34];	p[3] += buffer[35];
		p[4] += buffer[ 36];	p[5] += buffer[ 37];	p[6] += buffer[ 38];	p[7] += buffer[39];
		macro1();
		p[0] += buffer[ 40];	p[1] += buffer[ 41];	p[2] += buffer[ 42];	p[3] += buffer[43];
		p[4] += buffer[ 44];	p[5] += buffer[ 45];	p[6] += buffer[ 46];	p[7] += buffer[47];
		macro2();
		p[0] += buffer[ 48];	p[1] += buffer[ 49];	p[2] += buffer[ 50];	p[3] += buffer[51];
		p[4] += buffer[ 52];	p[5] += buffer[ 53];	p[6] += buffer[ 54];	p[7] += buffer[55];
		macro1();
		p[0] += buffer[ 56];	p[1] += buffer[ 57];	p[2] += buffer[ 58];	p[3] += buffer[59];
		p[4] += buffer[ 60];	p[5] += buffer[ 61];	p[6] += buffer[ 62];	p[7] += buffer[63];
		macro2();
		p[0] += buffer[ 64];	p[1] += buffer[ 65];	p[2] += buffer[ 66];	p[3] += buffer[67];
		p[4] += buffer[ 68];	p[5] += buffer[ 69];	p[6] += buffer[ 70];	p[7] += buffer[71];
		macro1();
		p[0] += buffer[ 72];	p[1] += buffer[ 73];	p[2] += buffer[ 74];	p[3] += buffer[75];
		p[4] += buffer[ 76];	p[5] += buffer[ 77];	p[6] += buffer[ 78];	p[7] += buffer[79];
		macro2();
		p[0] += buffer[ 80];	p[1] += buffer[ 81];	p[2] += buffer[ 82];	p[3] += buffer[83];
		p[4] += buffer[ 84];	p[5] += buffer[ 85];	p[6] += buffer[ 86];	p[7] += buffer[87];
		macro1();
		p[0] += buffer[ 88];	p[1] += buffer[ 89];	p[2] += buffer[ 90];	p[3] += buffer[91];
		p[4] += buffer[ 92];	p[5] += buffer[ 93];	p[6] += buffer[ 94];	p[7] += buffer[95];
		macro2();
		p[0] += buffer[ 96];	p[1] += buffer[ 97];	p[2] += buffer[ 98];	p[3] += buffer[99];
		p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		macro1();
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];
		p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		macro2();
		p[0]+= make_uint2(0xA9D5C3F4, 0xEABE394C);	p[1]+= make_uint2(0x1A75B523, 0x991112C7);
		p[2]+= make_uint2(0x660FCC33, 0xAE18A40B);	p[3]+= make_uint2(0x98173EC4, 0xCAB2076D);
		p[4]+= make_uint2(0x749C51CE, 0x4903ADFF);	p[5]+= make_uint2(0x9746DF43, 0xFD95DE39);
		p[6]+= make_uint2(0x27C79C0E, 0x8FD19341);	p[7]+= make_uint2(0xFF352CBF, 0x9A255629);
		macro1();
		p[0]+= make_uint2(0x1A75B523, 0x991112C7);	p[1]+= make_uint2(0x660FCC33, 0xAE18A40B);
		p[2]+= make_uint2(0x98173EC4, 0xCAB2076D);	p[3]+= make_uint2(0x749C51CE, 0x4903ADFF);
		p[4]+= make_uint2(0x9746DF03, 0x0D95DE39);	p[5]+= make_uint2(0x27C79C0E, 0x8FD19341);
		p[6]+= make_uint2(0xFF352CB1, 0x8A255629);	p[7]+= make_uint2(0xDF6CA7BF, 0x5DB62599);
		macro2();
		p[0] += vectorize(0xAE18A40B660FCC33);		p[1] += vectorize(0xcab2076d98173ec4);
		p[2] += vectorize(0x4903ADFF749C51CE);		p[3] += vectorize(0x0D95DE399746DF03);
		p[4] += vectorize(0x8FD1934127C79BCE);		p[5] += vectorize(0x8A255629FF352CB1);
		p[6] += vectorize(0x4DB62599DF6CA7F0);		p[7] += vectorize(0xEABE394CA9D5C3F4 + 16);
		macro1();
		p[0] += vectorize(0xcab2076d98173ec4);		p[1] += vectorize(0x4903ADFF749C51CE);
		p[2] += vectorize(0x0D95DE399746DF03);		p[3] += vectorize(0x8FD1934127C79BCE);
		p[4] += vectorize(0x9A255629FF352CB1);		p[5] += vectorize(0x4DB62599DF6CA7F0);
		p[6] += vectorize(0xEABE394CA9D5C3F4 + 0x0000000000000040);
		p[7] += vectorize(0x991112C71A75B523 + 17);
		macro2();
		p[0] += vectorize(0x4903ADFF749C51CE);		p[1] += vectorize(0x0D95DE399746DF03);
		p[2] += vectorize(0x8FD1934127C79BCE);		p[3] += vectorize(0x9A255629FF352CB1);
		p[4] += vectorize(0x5DB62599DF6CA7B0);		p[5] += vectorize(0xEABE394CA9D5C3F4 + 0x0000000000000040);
		p[6] += vectorize(0x891112C71A75B523);		p[7] += vectorize(0xAE18A40B660FCC33 + 18);

		#define h0 p[0]
		#define h1 p[1]
		#define h2 p[2]
		#define h3 p[3]
		#define h4 p[4]
		#define h5 p[5]
		#define h6 p[6]
		#define h7 p[7]

		h0 ^= h[0];	h1 ^= h[1];	h2 ^= h[2];	h3 ^= h[3];
		h4 ^= h[4];	h5 ^= h[5];	h6 ^= h[6];	h7 ^= h[7];

		uint2 skein_h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ vectorize(0x1BD11BDAA9FC1A22);

		uint2 hash64[8];

		hash64[5] = h5 + 8u;

		hash64[0] = h0 + h1;
		hash64[1] = ROL2(h1, 46) ^ hash64[0];
		hash64[2] = h2 + h3;
		hash64[3] = ROL2(h3, 36) ^ hash64[2];
		hash64[4] = h4 + hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] = (h6 + h7 + make_uint2(0,0xff000000));
		hash64[7] = ROL2(h7, 37) ^ hash64[6];
		hash64[2]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4]+= hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];

		hash64[0]+= h1; hash64[1]+= h2; hash64[2]+= h3; hash64[3]+= h4;
		hash64[4]+= h5;
		hash64[5]+= h6 + make_uint2(0,0xff000000);
		hash64[6]+= h7 + vectorize(0xff00000000000008);
		hash64[7]+= skein_h8 + 1u;
		macro3();
		hash64[0]+= h2; hash64[1]+= h3; hash64[2]+= h4; hash64[3]+= h5;
		hash64[4]+= h6;
		hash64[5]+= h7 + vectorize(0xff00000000000008);
		hash64[6]+= skein_h8 + 8u;
		hash64[7]+= h0 + 2u;
		macro4();
		hash64[0] = (hash64[0] + h3);			hash64[1] = (hash64[1] + h4);
		hash64[2] = (hash64[2] + h5);			hash64[3] = (hash64[3] + h6);
		hash64[4] = (hash64[4] + h7);			hash64[5] = (hash64[5] + skein_h8 + 8u);
		hash64[6] = (hash64[6] + h0 + make_uint2(0,0xff000000));
		hash64[7] = (hash64[7] + h1 + 3u);
		macro3();
		hash64[0] = (hash64[0] + h4);			hash64[1] = (hash64[1] + h5);
		hash64[2] = (hash64[2] + h6);			hash64[3] = (hash64[3] + h7);
		hash64[4] = (hash64[4] + skein_h8);		hash64[5] = (hash64[5] + h0 + make_uint2(0,0xff000000));
		hash64[6] = (hash64[6] + h1 + vectorize(0xff00000000000008));
		hash64[7] = (hash64[7] + h2 + 4u);
		macro4();
		hash64[0] = (hash64[0] + h5);			hash64[1] = (hash64[1] + h6);
		hash64[2] = (hash64[2] + h7);			hash64[3] = (hash64[3] + skein_h8);
		hash64[4] = (hash64[4] + h0);			hash64[5] = (hash64[5] + h1 + vectorize(0xff00000000000008));
		hash64[6] = (hash64[6] + h2 + 8u);		hash64[7] = (hash64[7] + h3 + 5u);
		macro3();
		hash64[0] = (hash64[0] + h6);			hash64[1] = (hash64[1] + h7);
		hash64[2] = (hash64[2] + skein_h8);		hash64[3] = (hash64[3] + h0);
		hash64[4] = (hash64[4] + h1);			hash64[5] = (hash64[5] + h2 + 8u);
		hash64[6] = (hash64[6] + h3 + make_uint2(0,0xff000000));
		hash64[7] = (hash64[7] + h4 + 6u);
		macro4();
		hash64[0] = (hash64[0] + h7);			hash64[1] = (hash64[1] + skein_h8);
		hash64[2] = (hash64[2] + h0);			hash64[3] = (hash64[3] + h1);
		hash64[4] = (hash64[4] + h2);			hash64[5] = (hash64[5] + h3 + make_uint2(0,0xff000000));
		hash64[6] = (hash64[6] + h4 + vectorize(0xff00000000000008));
		hash64[7] = (hash64[7] + h5 + 7u);
		macro3();
		hash64[0] = (hash64[0] + skein_h8);		hash64[1] = (hash64[1] + h0);
		hash64[2] = (hash64[2] + h1);			hash64[3] = (hash64[3] + h2);
		hash64[4] = (hash64[4] + h3);			hash64[5] = (hash64[5] + h4 + vectorize(0xff00000000000008));
		hash64[6] = (hash64[6] + h5 + 8u);		hash64[7] = (hash64[7] + h6 + 8u);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h0));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h1));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h2));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h3));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h4));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h5) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h6) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h7) + 9);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h1));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h2));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h3));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h4));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h5));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h6) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h7) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(skein_h8) + 10);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h2));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h3));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h4));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h5));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h6));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h7) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(skein_h8) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h0) + 11);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h3));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h4));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h5));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h6));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h7));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(skein_h8) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h0) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h1) + 12);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h4));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h5));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h6));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h7));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(skein_h8));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h0) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h1) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h2) + 13);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h5));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h6));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h7));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(skein_h8));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h0));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h1) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h2) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h3) + 14);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h6));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h7));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(skein_h8));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h0));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h1));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h2) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h3) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h4) + 15);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h7));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(skein_h8));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h0));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h1));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h2));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h3) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h4) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h5) + 16);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(skein_h8));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h0));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h1));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h2));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h3));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h4) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h5) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h6) + 17);

        //macro3();
        hash64[0] += hash64[1]; hash64[2] += hash64[3]; hash64[4] += hash64[5]; hash64[6] += hash64[7];
        hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
        hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
        hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
        hash64[7] = ROL24(hash64[7]) ^ hash64[6];
        hash64[2] += hash64[1]; hash64[4] += hash64[7]; hash64[6] += hash64[5]; hash64[0] += hash64[3];
        hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
        hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
        hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
        hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
        hash64[4] += hash64[1];
        hash64[6] += hash64[3];
        hash64[0] += hash64[5];
        hash64[2] += hash64[7];
        hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
        hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
        hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
        hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
        hash64[6] += hash64[1];
        hash64[0] += hash64[7];
        hash64[2] += hash64[5];
        hash64[4] += hash64[3];
        hash64[1] = ROL8(hash64[1]) ^ hash64[6];
        hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
        hash64[5] = ROR8(hash64[5]) ^ hash64[2];
        hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];

		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h0));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h1));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h2));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h3));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h4));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h5) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h6) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h7) + 18);

        // Phi2 special xor
        hash64[3] ^= hash64[7];

		if(devectorize(hash64[3]) <= target){
			uint32_t tmp = atomicExch(&resNonce[0], startNonce + thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;		
		}

		#undef h0
		#undef h1
		#undef h2
		#undef h3
		#undef h4
		#undef h5
		#undef h6
		#undef h7
	}
}

__host__
void cuda_phi2_skein512_cpu_hash_64f(const uint32_t threads, uint32_t *d_hash, const uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target) {

	uint32_t tpb = PHI2SKEIN512_TPB64F;

    const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	cuda_phi2_skein512_gpu_hash_64_final<<<grid, block >>>(threads, d_hash, startNonce, d_resNonce, target);

}

#pragma endregion
