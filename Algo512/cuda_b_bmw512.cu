#include <memory.h>

#include "cuda_helper.h"

#include "./cuda_b_bmw512.h"

#define BMW512_TPB80 128
#define BMW512_TPB64 256
#define BMW512_TPB64F 128


#pragma region macros

#define CONST_EXP2(i) \
	q[i+0] + ROL2(q[i+1], 5)  + q[i+2] + ROL2(q[i+3], 11) + \
	q[i+4] + ROL2(q[i+5], 27) + q[i+6] + SWAPUINT2(q[i+7]) + \
	q[i+8] + ROL2(q[i+9], 37) + q[i+10] + ROL2(q[i+11], 43) + \
	q[i+12] + ROL2(q[i+13], 53) + (SHR2(q[i+14],1) ^ q[i+14]) + (SHR2(q[i+15],2) ^ q[i+15])

#pragma endregion

#pragma region inlines

__device__
void bmw_compression512_64_first(uint2 *msg, uint2 *hash) {
	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[5] ^ hash[5]) - (msg[7] ^ hash[7]) + (hash[10]) + (hash[13]) + (hash[14]);
	q[0] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[1];

	tmp = (msg[6] ^ hash[6]) - (msg[8] ^ hash[8]) + (hash[11]) + (hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[2];
	tmp = (msg[0] ^ hash[0]) + (msg[7] ^ hash[7]) + (hash[9]) - (hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[3];
	tmp = (msg[0] ^ hash[0]) - (msg[1] ^ hash[1]) + (msg[8] ^ hash[8]) - (hash[10]) + (hash[13]);
	q[3] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[4];
	tmp = (msg[1] ^ hash[1]) + (msg[2] ^ hash[2]) + (hash[9]) - (hash[11]) - (hash[14]);
	q[4] = (SHR2(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[3] ^ hash[3]) - (msg[2] ^ hash[2]) + (hash[10]) - (hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[6];
	tmp = (msg[4] ^ hash[4]) - (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) - (hash[11]) + (hash[13]);
	q[6] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[7];
	tmp = (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[5] ^ hash[5]) - (hash[12]) - (hash[14]);
	q[7] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[8];

	tmp = (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) - (msg[6] ^ hash[6]) + (hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[9];
	tmp = (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) + (msg[6] ^ hash[6]) - (msg[7] ^ hash[7]) + (hash[14]);
	q[9] = (SHR2(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[8] ^ hash[8]) - (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[7] ^ hash[7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[11];
	tmp = (msg[8] ^ hash[8]) - (msg[0] ^ hash[0]) - (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) + (hash[9]);
	q[11] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[12];
	tmp = (msg[1] ^ hash[1]) + (msg[3] ^ hash[3]) - (msg[6] ^ hash[6]) - (hash[9]) + (hash[10]);
	q[12] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[13];
	tmp = (msg[2] ^ hash[2]) + (msg[4] ^ hash[4]) + (msg[7] ^ hash[7]) + (hash[10]) + (hash[11]);
	q[13] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[14];
	tmp = (msg[3] ^ hash[3]) - (msg[5] ^ hash[5]) + (msg[8] ^ hash[8]) - (hash[11]) - (hash[12]);
	q[14] = (SHR2(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[4] ^ hash[4]) - (msg[6] ^ hash[6]) - (hash[9]) + (hash[13]);
	q[15] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[0];

	q[0 + 16] =
		(SHR2(q[0], 1) ^ SHL2(q[0], 2) ^ ROL2(q[0], 13) ^ ROL2(q[0], 43)) +
		(SHR2(q[0 + 1], 2) ^ SHL2(q[0 + 1], 1) ^ ROL2(q[0 + 1], 19) ^ ROL2(q[0 + 1], 53)) +
		(SHR2(q[0 + 2], 2) ^ SHL2(q[0 + 2], 2) ^ ROL2(q[0 + 2], 28) ^ ROL2(q[0 + 2], 59)) +
		(SHR2(q[0 + 3], 1) ^ SHL2(q[0 + 3], 3) ^ ROL2(q[0 + 3], 4) ^ ROL2(q[0 + 3], 37)) +
		(SHR2(q[0 + 4], 1) ^ SHL2(q[0 + 4], 2) ^ ROL2(q[0 + 4], 13) ^ ROL2(q[0 + 4], 43)) +
		(SHR2(q[0 + 5], 2) ^ SHL2(q[0 + 5], 1) ^ ROL2(q[0 + 5], 19) ^ ROL2(q[0 + 5], 53)) +
		(SHR2(q[0 + 6], 2) ^ SHL2(q[0 + 6], 2) ^ ROL2(q[0 + 6], 28) ^ ROL2(q[0 + 6], 59)) +
		(SHR2(q[0 + 7], 1) ^ SHL2(q[0 + 7], 3) ^ ROL2(q[0 + 7], 4) ^ ROL2(q[0 + 7], 37)) +
		(SHR2(q[0 + 8], 1) ^ SHL2(q[0 + 8], 2) ^ ROL2(q[0 + 8], 13) ^ ROL2(q[0 + 8], 43)) +
		(SHR2(q[0 + 9], 2) ^ SHL2(q[0 + 9], 1) ^ ROL2(q[0 + 9], 19) ^ ROL2(q[0 + 9], 53)) +
		(SHR2(q[0 + 10], 2) ^ SHL2(q[0 + 10], 2) ^ ROL2(q[0 + 10], 28) ^ ROL2(q[0 + 10], 59)) +
		(SHR2(q[0 + 11], 1) ^ SHL2(q[0 + 11], 3) ^ ROL2(q[0 + 11], 4) ^ ROL2(q[0 + 11], 37)) +
		(SHR2(q[0 + 12], 1) ^ SHL2(q[0 + 12], 2) ^ ROL2(q[0 + 12], 13) ^ ROL2(q[0 + 12], 43)) +
		(SHR2(q[0 + 13], 2) ^ SHL2(q[0 + 13], 1) ^ ROL2(q[0 + 13], 19) ^ ROL2(q[0 + 13], 53)) +
		(SHR2(q[0 + 14], 2) ^ SHL2(q[0 + 14], 2) ^ ROL2(q[0 + 14], 28) ^ ROL2(q[0 + 14], 59)) +
		(SHR2(q[0 + 15], 1) ^ SHL2(q[0 + 15], 3) ^ ROL2(q[0 + 15], 4) ^ ROL2(q[0 + 15], 37)) +
		((make_uint2(0x55555550ul,0x55555555) + ROL2(msg[0], 0 + 1) +
		ROL2(msg[0 + 3], 0 + 4)) ^ hash[0 + 7]);

	q[1 + 16] =
		(SHR2(q[1], 1) ^ SHL2(q[1], 2) ^ ROL2(q[1], 13) ^ ROL2(q[1], 43)) +
		(SHR2(q[1 + 1], 2) ^ SHL2(q[1 + 1], 1) ^ ROL2(q[1 + 1], 19) ^ ROL2(q[1 + 1], 53)) +
		(SHR2(q[1 + 2], 2) ^ SHL2(q[1 + 2], 2) ^ ROL2(q[1 + 2], 28) ^ ROL2(q[1 + 2], 59)) +
		(SHR2(q[1 + 3], 1) ^ SHL2(q[1 + 3], 3) ^ ROL2(q[1 + 3], 4) ^ ROL2(q[1 + 3], 37)) +
		(SHR2(q[1 + 4], 1) ^ SHL2(q[1 + 4], 2) ^ ROL2(q[1 + 4], 13) ^ ROL2(q[1 + 4], 43)) +
		(SHR2(q[1 + 5], 2) ^ SHL2(q[1 + 5], 1) ^ ROL2(q[1 + 5], 19) ^ ROL2(q[1 + 5], 53)) +
		(SHR2(q[1 + 6], 2) ^ SHL2(q[1 + 6], 2) ^ ROL2(q[1 + 6], 28) ^ ROL2(q[1 + 6], 59)) +
		(SHR2(q[1 + 7], 1) ^ SHL2(q[1 + 7], 3) ^ ROL2(q[1 + 7], 4) ^ ROL2(q[1 + 7], 37)) +
		(SHR2(q[1 + 8], 1) ^ SHL2(q[1 + 8], 2) ^ ROL2(q[1 + 8], 13) ^ ROL2(q[1 + 8], 43)) +
		(SHR2(q[1 + 9], 2) ^ SHL2(q[1 + 9], 1) ^ ROL2(q[1 + 9], 19) ^ ROL2(q[1 + 9], 53)) +
		(SHR2(q[1 + 10], 2) ^ SHL2(q[1 + 10], 2) ^ ROL2(q[1 + 10], 28) ^ ROL2(q[1 + 10], 59)) +
		(SHR2(q[1 + 11], 1) ^ SHL2(q[1 + 11], 3) ^ ROL2(q[1 + 11], 4) ^ ROL2(q[1 + 11], 37)) +
		(SHR2(q[1 + 12], 1) ^ SHL2(q[1 + 12], 2) ^ ROL2(q[1 + 12], 13) ^ ROL2(q[1 + 12], 43)) +
		(SHR2(q[1 + 13], 2) ^ SHL2(q[1 + 13], 1) ^ ROL2(q[1 + 13], 19) ^ ROL2(q[1 + 13], 53)) +
		(SHR2(q[1 + 14], 2) ^ SHL2(q[1 + 14], 2) ^ ROL2(q[1 + 14], 28) ^ ROL2(q[1 + 14], 59)) +
		(SHR2(q[1 + 15], 1) ^ SHL2(q[1 + 15], 3) ^ ROL2(q[1 + 15], 4) ^ ROL2(q[1 + 15], 37)) +
		((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL2(msg[1], 1 + 1) +
		ROL2(msg[1 + 3], 1 + 4)) ^ hash[1 + 7]);

	q[2 + 16] = CONST_EXP2(2) +
		((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL2(msg[2], 2 + 1) +
		ROL2(msg[2 + 3], 2 + 4) - ROL2(msg[2 + 10], 2 + 11)) ^ hash[2 + 7]);
	q[3 + 16] = CONST_EXP2(3) +
		((make_uint2(0x5555554F, 0x65555555) + ROL2(msg[3], 3 + 1) +
		ROL2(msg[3 + 3], 3 + 4) - ROL2(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) +ROL2(msg[4], 4 + 1) +
		ROL2(msg[4 + 3], 4 + 4) - ROL2(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL2(msg[5], 5 + 1) +
		ROL2(msg[5 + 3], 5 + 4) - ROL2(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);

	#pragma unroll 3
	for (int i = 6; i<9; i++) {
		q[i + 16] = CONST_EXP2(i) +
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROL2(msg[i], i + 1) -
			ROL2(msg[i - 6], (i - 6) + 1)) ^ hash[i + 7]);
	}

	#pragma unroll 4
	for (int i = 9; i<13; i++) {
		q[i + 16] = CONST_EXP2(i) +
			((vectorize((i + 16)*(0x0555555555555555ull)) +
			ROL2(msg[i + 3], i + 4) - ROL2(msg[i - 6], (i - 6) + 1)) ^ hash[i - 9]);
	}

	q[13 + 16] = CONST_EXP2(13) +
		((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL2(msg[13], 13 + 1) +
		ROL2(msg[13 - 13], (13 - 13) + 1) - ROL2(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL2(msg[14], 14 + 1) +
		ROL2(msg[14 - 13], (14 - 13) + 1) - ROL2(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((make_uint2(0x5555554B, 0xA5555555) + ROL2(msg[15], 15 + 1) +
		ROL2(msg[15 - 13], (15 - 13) + 1) - ROL2(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);


	uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL2(XH64, 5) ^ SHR2(q[16], 5) ^ msg[0]) + (XL64 ^ q[24] ^ q[0]);
	hash[1] = (SHR2(XH64, 7) ^ SHL2(q[17], 8) ^ msg[1]) + (XL64 ^ q[25] ^ q[1]);
	hash[2] = (SHR2(XH64, 5) ^ SHL2(q[18], 5) ^ msg[2]) + (XL64 ^ q[26] ^ q[2]);
	hash[3] = (SHR2(XH64, 1) ^ SHL2(q[19], 5) ^ msg[3]) + (XL64 ^ q[27] ^ q[3]);
	hash[4] = (SHR2(XH64, 3) ^ q[20] ^ msg[4]) + (XL64 ^ q[28] ^ q[4]);
	hash[5] = (SHL2(XH64, 6) ^ SHR2(q[21], 6) ^ msg[5]) + (XL64 ^ q[29] ^ q[5]);
	hash[6] = (SHR2(XH64, 4) ^ SHL2(q[22], 6) ^ msg[6]) + (XL64 ^ q[30] ^ q[6]);
	hash[7] = (SHR2(XH64, 11) ^ SHL2(q[23], 2) ^ msg[7]) + (XL64 ^ q[31] ^ q[7]);

	hash[8] =  ROL2(hash[4], 9)  + (XH64 ^ q[24] ^ msg[8]) + (SHL2(XL64, 8) ^ q[23] ^ q[8]);
	hash[9] =  ROL2(hash[5], 10) + (XH64 ^ q[25]) + (SHR2(XL64, 6) ^ q[16] ^ q[9]);
	hash[10] = ROL2(hash[6], 11) + (XH64 ^ q[26]) + (SHL2(XL64, 6) ^ q[17] ^ q[10]);
	hash[11] = ROL2(hash[7], 12) + (XH64 ^ q[27]) + (SHL2(XL64, 4) ^ q[18] ^ q[11]);
	hash[12] = ROL2(hash[0], 13) + (XH64 ^ q[28]) + (SHR2(XL64, 3) ^ q[19] ^ q[12]);
	hash[13] = ROL2(hash[1], 14) + (XH64 ^ q[29]) + (SHR2(XL64, 4) ^ q[20] ^ q[13]);
	hash[14] = ROL2(hash[2], 15) + (XH64 ^ q[30]) + (SHR2(XL64, 7) ^ q[21] ^ q[14]);
	hash[15] = ROL2(hash[3], 16) + (XH64 ^ q[31] ^ msg[15]) + (SHR2(XL64, 2) ^ q[22] ^ q[15]);
}

__device__ __forceinline__
void bmw_compression512(uint2 *msg, uint2 *hash)
{
	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[ 5] ^ hash[ 5]) - (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
	q[0] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp,  4) ^ ROL2(tmp, 37)) + hash[1];
	tmp = (msg[ 6] ^ hash[ 6]) - (msg[ 8] ^ hash[ 8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[2];
	tmp = (msg[ 0] ^ hash[ 0]) + (msg[ 7] ^ hash[ 7]) + (msg[ 9] ^ hash[ 9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[3];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 1] ^ hash[ 1]) + (msg[ 8] ^ hash[ 8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
	q[3] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[4];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 2] ^ hash[ 2]) + (msg[ 9] ^ hash[ 9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
	q[4] = (SHR2(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 2] ^ hash[ 2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp,  4) ^ ROL2(tmp, 37)) + hash[6];
	tmp = (msg[ 4] ^ hash[ 4]) - (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
	q[6] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[7];
	tmp = (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 5] ^ hash[ 5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
	q[7] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[8];
	tmp = (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) - (msg[ 6] ^ hash[ 6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[9];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) + (msg[ 6] ^ hash[ 6]) - (msg[ 7] ^ hash[ 7]) + (msg[14] ^ hash[14]);
	q[9] = (SHR2(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 7] ^ hash[ 7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp,  4) ^ ROL2(tmp, 37)) + hash[11];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 0] ^ hash[ 0]) - (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) + (msg[ 9] ^ hash[ 9]);
	q[11] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[12];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 3] ^ hash[ 3]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[10] ^ hash[10]);
	q[12] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[13];
	tmp = (msg[ 2] ^ hash[ 2]) + (msg[ 4] ^ hash[ 4]) + (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
	q[13] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[14];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 5] ^ hash[ 5]) + (msg[ 8] ^ hash[ 8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
	q[14] = (SHR2(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[ 4] ^ hash[ 4]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[13] ^ hash[13]);
	q[15] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[0];

	q[0+16] =
		(SHR2(q[0], 1) ^ SHL2(q[0], 2) ^ ROL2(q[0], 13) ^ ROL2(q[0], 43)) +
		(SHR2(q[0+1], 2) ^ SHL2(q[0+1], 1) ^ ROL2(q[0+1], 19) ^ ROL2(q[0+1], 53)) +
		(SHR2(q[0+2], 2) ^ SHL2(q[0+2], 2) ^ ROL2(q[0+2], 28) ^ ROL2(q[0+2], 59)) +
		(SHR2(q[0+3], 1) ^ SHL2(q[0+3], 3) ^ ROL2(q[0+3],  4) ^ ROL2(q[0+3], 37)) +
		(SHR2(q[0+4], 1) ^ SHL2(q[0+4], 2) ^ ROL2(q[0+4], 13) ^ ROL2(q[0+4], 43)) +
		(SHR2(q[0+5], 2) ^ SHL2(q[0+5], 1) ^ ROL2(q[0+5], 19) ^ ROL2(q[0+5], 53)) +
		(SHR2(q[0+6], 2) ^ SHL2(q[0+6], 2) ^ ROL2(q[0+6], 28) ^ ROL2(q[0+6], 59)) +
		(SHR2(q[0+7], 1) ^ SHL2(q[0+7], 3) ^ ROL2(q[0+7],  4) ^ ROL2(q[0+7], 37)) +
		(SHR2(q[0+8], 1) ^ SHL2(q[0+8], 2) ^ ROL2(q[0+8], 13) ^ ROL2(q[0+8], 43)) +
		(SHR2(q[0+9], 2) ^ SHL2(q[0+9], 1) ^ ROL2(q[0+9], 19) ^ ROL2(q[0+9], 53)) +
		(SHR2(q[0+10], 2) ^ SHL2(q[0+10], 2) ^ ROL2(q[0+10], 28) ^ ROL2(q[0+10], 59)) +
		(SHR2(q[0+11], 1) ^ SHL2(q[0+11], 3) ^ ROL2(q[0+11],  4) ^ ROL2(q[0+11], 37)) +
		(SHR2(q[0+12], 1) ^ SHL2(q[0+12], 2) ^ ROL2(q[0+12], 13) ^ ROL2(q[0+12], 43)) +
		(SHR2(q[0+13], 2) ^ SHL2(q[0+13], 1) ^ ROL2(q[0+13], 19) ^ ROL2(q[0+13], 53)) +
		(SHR2(q[0+14], 2) ^ SHL2(q[0+14], 2) ^ ROL2(q[0+14], 28) ^ ROL2(q[0+14], 59)) +
		(SHR2(q[0+15], 1) ^ SHL2(q[0+15], 3) ^ ROL2(q[0+15],  4) ^ ROL2(q[0+15], 37)) +
		((make_uint2(0x55555550ul, 0x55555555) + ROL2(msg[0], 0 + 1) +
		ROL2(msg[0+3], 0+4) - ROL2(msg[0+10], 0+11) ) ^ hash[0+7]);

	q[1 + 16] =
		(SHR2(q[1], 1) ^ SHL2(q[1], 2) ^ ROL2(q[1], 13) ^ ROL2(q[1], 43)) +
		(SHR2(q[1 + 1], 2) ^ SHL2(q[1 + 1], 1) ^ ROL2(q[1 + 1], 19) ^ ROL2(q[1 + 1], 53)) +
		(SHR2(q[1 + 2], 2) ^ SHL2(q[1 + 2], 2) ^ ROL2(q[1 + 2], 28) ^ ROL2(q[1 + 2], 59)) +
		(SHR2(q[1 + 3], 1) ^ SHL2(q[1 + 3], 3) ^ ROL2(q[1 + 3], 4) ^ ROL2(q[1 + 3], 37)) +
		(SHR2(q[1 + 4], 1) ^ SHL2(q[1 + 4], 2) ^ ROL2(q[1 + 4], 13) ^ ROL2(q[1 + 4], 43)) +
		(SHR2(q[1 + 5], 2) ^ SHL2(q[1 + 5], 1) ^ ROL2(q[1 + 5], 19) ^ ROL2(q[1 + 5], 53)) +
		(SHR2(q[1 + 6], 2) ^ SHL2(q[1 + 6], 2) ^ ROL2(q[1 + 6], 28) ^ ROL2(q[1 + 6], 59)) +
		(SHR2(q[1 + 7], 1) ^ SHL2(q[1 + 7], 3) ^ ROL2(q[1 + 7], 4) ^ ROL2(q[1 + 7], 37)) +
		(SHR2(q[1 + 8], 1) ^ SHL2(q[1 + 8], 2) ^ ROL2(q[1 + 8], 13) ^ ROL2(q[1 + 8], 43)) +
		(SHR2(q[1 + 9], 2) ^ SHL2(q[1 + 9], 1) ^ ROL2(q[1 + 9], 19) ^ ROL2(q[1 + 9], 53)) +
		(SHR2(q[1 + 10], 2) ^ SHL2(q[1 + 10], 2) ^ ROL2(q[1 + 10], 28) ^ ROL2(q[1 + 10], 59)) +
		(SHR2(q[1 + 11], 1) ^ SHL2(q[1 + 11], 3) ^ ROL2(q[1 + 11], 4) ^ ROL2(q[1 + 11], 37)) +
		(SHR2(q[1 + 12], 1) ^ SHL2(q[1 + 12], 2) ^ ROL2(q[1 + 12], 13) ^ ROL2(q[1 + 12], 43)) +
		(SHR2(q[1 + 13], 2) ^ SHL2(q[1 + 13], 1) ^ ROL2(q[1 + 13], 19) ^ ROL2(q[1 + 13], 53)) +
		(SHR2(q[1 + 14], 2) ^ SHL2(q[1 + 14], 2) ^ ROL2(q[1 + 14], 28) ^ ROL2(q[1 + 14], 59)) +
		(SHR2(q[1 + 15], 1) ^ SHL2(q[1 + 15], 3) ^ ROL2(q[1 + 15], 4) ^ ROL2(q[1 + 15], 37)) +
		((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL2(msg[1], 1 + 1) +
		ROL2(msg[1 + 3], 1 + 4) - ROL2(msg[1 + 10], 1 + 11)) ^ hash[1 + 7]);

	q[2 + 16] = CONST_EXP2(2) +
		((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL2(msg[2], 2 + 1) +
		ROL2(msg[2+3], 2+4) - ROL2(msg[2+10], 2+11) ) ^ hash[2+7]);
	q[3 + 16] = CONST_EXP2(3) +
		((make_uint2(0x5555554F, 0x65555555) + ROL2(msg[3], 3 + 1) +
		ROL2(msg[3 + 3], 3 + 4) - ROL2(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) + ROL2(msg[4], 4 + 1) +
		ROL2(msg[4 + 3], 4 + 4) - ROL2(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL2(msg[5], 5 + 1) +
		ROL2(msg[5 + 3], 5 + 4) - ROL2(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);
	q[6 + 16] = CONST_EXP2(6) +
		((make_uint2(0x5555554E, 0x75555555)+ ROL2(msg[6], 6 + 1) +
		ROL2(msg[6 + 3], 6 + 4) - ROL2(msg[6 - 6], (6 - 6) + 1)) ^ hash[6 + 7]);
	q[7 + 16] = CONST_EXP2(7) +
		((make_uint2(0xAAAAAAA3, 0x7AAAAAAA) + ROL2(msg[7], 7 + 1) +
		ROL2(msg[7 + 3], 7 + 4) - ROL2(msg[7 - 6], (7 - 6) + 1)) ^ hash[7 + 7]);
	q[8 + 16] = CONST_EXP2(8) +
		((make_uint2(0xFFFFFFF8, 0x7FFFFFFF) + ROL2(msg[8], 8 + 1) +
		ROL2(msg[8 + 3], 8 + 4) - ROL2(msg[8 - 6], (8 - 6) + 1)) ^ hash[8 + 7]);
	q[9 + 16] = CONST_EXP2(9) +
		((make_uint2(0x5555554D, 0x85555555) + ROL2(msg[9], 9 + 1) +
		ROL2(msg[9 + 3], 9 + 4) - ROL2(msg[9 - 6], (9 - 6) + 1)) ^ hash[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((make_uint2(0xAAAAAAA2, 0x8AAAAAAA) + ROL2(msg[10], 10 + 1) +
		ROL2(msg[10 + 3], 10 + 4) - ROL2(msg[10 - 6], (10 - 6) + 1)) ^ hash[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((make_uint2(0xFFFFFFF7, 0x8FFFFFFF) + ROL2(msg[11], 11 + 1) +
		ROL2(msg[11 + 3], 11 + 4) - ROL2(msg[11 - 6], (11 - 6) + 1)) ^ hash[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((make_uint2(0x5555554C, 0x95555555) + ROL2(msg[12], 12 + 1) +
		ROL2(msg[12 + 3], 12 + 4) - ROL2(msg[12 - 6], (12 - 6) + 1)) ^ hash[12 - 9]);
	q[13 + 16] = CONST_EXP2(13) +
		((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL2(msg[13], 13 + 1) +
		ROL2(msg[13 - 13], (13 - 13) + 1) - ROL2(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL2(msg[14], 14 + 1) +
		ROL2(msg[14 - 13], (14 - 13) + 1) - ROL2(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((make_uint2(0x5555554B, 0xA5555555) + ROL2(msg[15], 15 + 1) +
		ROL2(msg[15 - 13], (15 - 13) + 1) - ROL2(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);

	uint2 XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL2(XH64, 5) ^ SHR2(q[16],5) ^ msg[ 0]) + (XL64 ^ q[24] ^ q[ 0]);
	hash[1] = (SHR2(XH64, 7) ^ SHL2(q[17],8) ^ msg[ 1]) + (XL64 ^ q[25] ^ q[ 1]);
	hash[2] = (SHR2(XH64, 5) ^ SHL2(q[18],5) ^ msg[ 2]) + (XL64 ^ q[26] ^ q[ 2]);
	hash[3] = (SHR2(XH64, 1) ^ SHL2(q[19],5) ^ msg[ 3]) + (XL64 ^ q[27] ^ q[ 3]);
	hash[4] = (SHR2(XH64, 3) ^     q[20]    ^ msg[ 4]) + (XL64 ^ q[28] ^ q[ 4]);
	hash[5] = (SHL2(XH64, 6) ^ SHR2(q[21],6) ^ msg[ 5]) + (XL64 ^ q[29] ^ q[ 5]);
	hash[6] = (SHR2(XH64, 4) ^ SHL2(q[22],6) ^ msg[ 6]) + (XL64 ^ q[30] ^ q[ 6]);
	hash[7] = (SHR2(XH64,11) ^ SHL2(q[23],2) ^ msg[ 7]) + (XL64 ^ q[31] ^ q[ 7]);

	hash[ 8] = ROL2(hash[4], 9) + (XH64 ^ q[24] ^ msg[ 8]) + (SHL2(XL64,8) ^ q[23] ^ q[ 8]);
	hash[ 9] = ROL2(hash[5],10) + (XH64 ^ q[25] ^ msg[ 9]) + (SHR2(XL64,6) ^ q[16] ^ q[ 9]);
	hash[10] = ROL2(hash[6],11) + (XH64 ^ q[26] ^ msg[10]) + (SHL2(XL64,6) ^ q[17] ^ q[10]);
	hash[11] = ROL2(hash[7],12) + (XH64 ^ q[27] ^ msg[11]) + (SHL2(XL64,4) ^ q[18] ^ q[11]);
	hash[12] = ROL2(hash[0],13) + (XH64 ^ q[28] ^ msg[12]) + (SHR2(XL64,3) ^ q[19] ^ q[12]);
	hash[13] = ROL2(hash[1],14) + (XH64 ^ q[29] ^ msg[13]) + (SHR2(XL64,4) ^ q[20] ^ q[13]);
	hash[14] = ROL2(hash[2],15) + (XH64 ^ q[30] ^ msg[14]) + (SHR2(XL64,7) ^ q[21] ^ q[14]);
	hash[15] = ROL2(hash[3],16) + (XH64 ^ q[31] ^ msg[15]) + (SHR2(XL64, 2) ^ q[22] ^ q[15]);
}

#pragma endregion


#pragma region BMW512_80

__constant__ uint64_t c_paddedMessage80[16]; // padded message (80 bytes + padding)

__host__
void cuda_base_bmw512_cpu_setBlock_80(void *pdata) {

	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	uint64_t *message = (uint64_t*)PaddedMessage;
	message[10] = SPH_C64(0x80);
	message[15] = SPH_C64(640);
	cudaMemcpyToSymbol(c_paddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__global__
__launch_bounds__(BMW512_TPB80, 3)
void cuda_base_bmw512_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint64_t *g_hash) {

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

        // Init
		uint2 h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};
		// Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
		// BMW arbeitet mit 128 Byte!!!
		uint2 message[16];
        #pragma unroll 16
		for(int i=0;i<16;i++)
			message[i] = vectorize(c_paddedMessage80[i]);

		// die Nounce durch die thread-spezifische ersetzen
		message[9].y = cuda_swab32(startNounce + thread);	//REPLACE_HIDWORD(message[9], cuda_swab32(nounce));

		// Compression 1
		bmw_compression512(message, h);

        #pragma unroll 16
		for(int i=0;i<16;i++)
			message[i] = make_uint2(0xaaaaaaa0+i,0xaaaaaaaa);


		bmw_compression512(h, message);

		// fertig
		uint64_t *outpHash = &g_hash[thread * 8];

        #pragma unroll 8
		for(int i=0;i<8;i++)
			outpHash[i] = devectorize(message[i+8]);
	}
}

__host__
void cuda_base_bmw512_cpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = BMW512_TPB80;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

    cuda_base_bmw512_gpu_hash_80<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
}

#pragma endregion


#pragma region BMW512_64

__global__
//__launch_bounds__(BMW512_TPB64, 2)
void cuda_base_bmw512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash) {

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {

        uint64_t *pHash = &g_hash[thread * 8];

		// Init
		const uint2 __align__(16) hash[16] = {
			{ 0x84858687UL, 0x80818283UL }, { 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL }, { 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL }, { 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL }, { 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL }, { 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL }, { 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL }, { 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL }, { 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};

        const uint64_t __align__(16) HZero[16] = {
            0x8081828384858687, 0x88898A8B8C8D8E8F,
            0x9091929394959697, 0x98999A9B9C9D9E9F,
            0xA0A1A2A3A4A5A6A7, 0xA8A9AAABACADAEAF,
            0xB0B1B2B3B4B5B6B7, 0xB8B9BABBBCBDBEBF,
            0xC0C1C2C3C4C5C6C7 ^ 0x80, 0xC8C9CACBCCCDCECF,
            0xD0D1D2D3D4D5D6D7, 0xD8D9DADBDCDDDEDF,
            0xE0E1E2E3E4E5E6E7, 0xE8E9EAEBECEDEEEF,
            0xF0F1F2F3F4F5F6F7, 0xF8F9FAFBFCFDFEFF ^ 512
        };


        uint64_t inMsg[8];
        #pragma unroll
        for (int i = 0; i < 8; i+=2) {
            AS_UINT4(&inMsg[i]) = AS_UINT4(&pHash[i]);
        }

		// Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
		// BMW arbeitet mit 128 Byte!!!
		uint2 msg[16];
		#pragma unroll
		for(int i=0;i<8;i++)
			msg[i] = vectorize(inMsg[i]);

		// Padding einfügen (Byteorder?!?)
		msg[8] = make_uint2(0x80,0);
        msg[9] = msg[10] = msg[11] = msg[12] = msg[13] = msg[14] = make_uint2(0,0);
		// Länge (in Bits, d.h. 64 Byte * 8 = 512 Bits
		msg[15] = make_uint2(512,0);

		// Compression 1
		//bmw_compression512_64_first(msg, hash);

        uint2 q[32];
        uint2 tmp;
        uint64_t temp;
        uint64_t xMsg[8];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            xMsg[i] = inMsg[i] ^ HZero[i];
        }

	    //tmp = vectorize(xMsg[5] - xMsg[7] + HZero[10] + HZero[13] + HZero[14]);
	    temp = xMsg[5] - xMsg[7] + HZero[10] + HZero[13] + HZero[14];
	    //q[0] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[1];
	    q[0] = vectorize((shr_t64(temp, 1) ^ shl_t64(temp, 3) ^ ROTL64(temp, 4) ^ ROTL64(temp, 37)) + HZero[1]);
	    tmp = vectorize(xMsg[6] - HZero[8] + HZero[11] + HZero[14] - HZero[15]);
	    q[1] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[2];
	    tmp = vectorize(xMsg[0] + xMsg[7] + HZero[9] - HZero[12] + HZero[15]);
	    q[2] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[3];

	    q[16] = (SHR2(q[0], 1) ^ SHL2(q[0], 2) ^ ROL2(q[0], 13) ^ ROL2(q[0], 43))
              + (SHR2(q[1], 2) ^ SHL2(q[1], 1) ^ ROL2(q[1], 19) ^ ROL2(q[1], 53))
              + (SHR2(q[2], 2) ^ SHL2(q[2], 2) ^ ROL2(q[2], 28) ^ ROL2(q[2], 59));
	    q[17] = (SHR2(q[1], 1) ^ SHL2(q[1], 2) ^ ROL2(q[1], 13) ^ ROL2(q[1], 43))
              + (SHR2(q[2], 2) ^ SHL2(q[2], 1) ^ ROL2(q[2], 19) ^ ROL2(q[2], 53));

	    tmp = vectorize(xMsg[0] - xMsg[1] + HZero[8] - HZero[10] + HZero[13]);
	    q[3] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[4];
	    tmp = vectorize(xMsg[1] + xMsg[2] + HZero[9] - HZero[11] - HZero[14]);
	    q[4] = (SHR2(tmp, 1) ^ tmp) + hash[5];

        q[16] += (SHR2(q[3], 1) ^ SHL2(q[3], 3) ^ ROL2(q[3], 4) ^ ROL2(q[3], 37))
            + (SHR2(q[4], 1) ^ SHL2(q[4], 2) ^ ROL2(q[4], 13) ^ ROL2(q[4], 43));
        q[17] += (SHR2(q[3], 2) ^ SHL2(q[3], 2) ^ ROL2(q[3], 28) ^ ROL2(q[3], 59))
            + (SHR2(q[4], 1) ^ SHL2(q[4], 3) ^ ROL2(q[4], 4) ^ ROL2(q[4], 37));

	    tmp = vectorize(xMsg[3] - xMsg[2] + HZero[10] - HZero[12] + HZero[15]);
	    q[5] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[6];
	    tmp = vectorize(xMsg[4] - xMsg[0] - xMsg[3] - HZero[11] + HZero[13]);
	    q[6] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[7];

        q[16] += (SHR2(q[5], 2) ^ SHL2(q[5], 1) ^ ROL2(q[5], 19) ^ ROL2(q[5], 53))
            + (SHR2(q[6], 2) ^ SHL2(q[6], 2) ^ ROL2(q[6], 28) ^ ROL2(q[6], 59));
        q[17] += (SHR2(q[5], 1) ^ SHL2(q[5], 2) ^ ROL2(q[5], 13) ^ ROL2(q[5], 43))
            + (SHR2(q[6], 2) ^ SHL2(q[6], 1) ^ ROL2(q[6], 19) ^ ROL2(q[6], 53));

	    temp = xMsg[1] - xMsg[4] - xMsg[5] - HZero[12] - HZero[14];
	    q[7] = vectorize((shr_t64(temp, 2) ^ shl_t64(temp, 1) ^ ROTL64(temp, 19) ^ ROTL64(temp, 53)) + 0xC0C1C2C3C4C5C6C7);

	    tmp = vectorize(xMsg[2] - xMsg[5] - xMsg[6] + HZero[13] - HZero[15]);
	    q[8] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[9];
	    tmp = vectorize(xMsg[0] - xMsg[3] + xMsg[6] - xMsg[7] + HZero[14]);
	    q[9] = (SHR2(tmp, 1) ^ tmp) + hash[10];
	    tmp = vectorize(HZero[8] - xMsg[1] - xMsg[4] - xMsg[7] + HZero[15]);
	    q[10] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[11];
	    tmp = vectorize(HZero[8] - xMsg[0] - xMsg[2] - xMsg[5] + HZero[9]);
	    q[11] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + hash[12];
	    tmp = vectorize(xMsg[1] + xMsg[3] - xMsg[6] - HZero[9] + HZero[10]);
	    q[12] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + hash[13];
	    tmp = vectorize(xMsg[2] + xMsg[4] + xMsg[7] + HZero[10] + HZero[11]);
	    q[13] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + hash[14];
	    temp = xMsg[3] - xMsg[5] + HZero[8] - HZero[11] - HZero[12];
	    q[14] = vectorize((shr_t64(temp, 1) ^ temp) + 0xF8F9FAFBFCFDFEFF);
	    tmp = vectorize(HZero[12] - xMsg[4] - xMsg[6] - HZero[9] + HZero[13]);
	    q[15] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + hash[0];

	    q[0 + 16] +=
		    //(SHR2(q[0], 1) ^ SHL2(q[0], 2) ^ ROL2(q[0], 13) ^ ROL2(q[0], 43)) +
		    //(SHR2(q[0 + 1], 2) ^ SHL2(q[0 + 1], 1) ^ ROL2(q[0 + 1], 19) ^ ROL2(q[0 + 1], 53)) +
		    //(SHR2(q[0 + 2], 2) ^ SHL2(q[0 + 2], 2) ^ ROL2(q[0 + 2], 28) ^ ROL2(q[0 + 2], 59)) +
		    //(SHR2(q[0 + 3], 1) ^ SHL2(q[0 + 3], 3) ^ ROL2(q[0 + 3], 4) ^ ROL2(q[0 + 3], 37)) +
		    //(SHR2(q[0 + 4], 1) ^ SHL2(q[0 + 4], 2) ^ ROL2(q[0 + 4], 13) ^ ROL2(q[0 + 4], 43)) +
		    //(SHR2(q[0 + 5], 2) ^ SHL2(q[0 + 5], 1) ^ ROL2(q[0 + 5], 19) ^ ROL2(q[0 + 5], 53)) +
		    //(SHR2(q[0 + 6], 2) ^ SHL2(q[0 + 6], 2) ^ ROL2(q[0 + 6], 28) ^ ROL2(q[0 + 6], 59)) +
		    (SHR2(q[0 + 7], 1) ^ SHL2(q[0 + 7], 3) ^ ROL2(q[0 + 7], 4) ^ ROL2(q[0 + 7], 37)) +
		    (SHR2(q[0 + 8], 1) ^ SHL2(q[0 + 8], 2) ^ ROL2(q[0 + 8], 13) ^ ROL2(q[0 + 8], 43)) +
		    (SHR2(q[0 + 9], 2) ^ SHL2(q[0 + 9], 1) ^ ROL2(q[0 + 9], 19) ^ ROL2(q[0 + 9], 53)) +
		    (SHR2(q[0 + 10], 2) ^ SHL2(q[0 + 10], 2) ^ ROL2(q[0 + 10], 28) ^ ROL2(q[0 + 10], 59)) +
		    (SHR2(q[0 + 11], 1) ^ SHL2(q[0 + 11], 3) ^ ROL2(q[0 + 11], 4) ^ ROL2(q[0 + 11], 37)) +
		    (SHR2(q[0 + 12], 1) ^ SHL2(q[0 + 12], 2) ^ ROL2(q[0 + 12], 13) ^ ROL2(q[0 + 12], 43)) +
		    (SHR2(q[0 + 13], 2) ^ SHL2(q[0 + 13], 1) ^ ROL2(q[0 + 13], 19) ^ ROL2(q[0 + 13], 53)) +
		    (SHR2(q[0 + 14], 2) ^ SHL2(q[0 + 14], 2) ^ ROL2(q[0 + 14], 28) ^ ROL2(q[0 + 14], 59)) +
		    (SHR2(q[0 + 15], 1) ^ SHL2(q[0 + 15], 3) ^ ROL2(q[0 + 15], 4) ^ ROL2(q[0 + 15], 37)) +
		    ((make_uint2(0x55555550ul,0x55555555) + ROL2(msg[0], 0 + 1) +
		    ROL2(msg[0 + 3], 0 + 4)) ^ hash[0 + 7]);

	    q[1 + 16] +=
		    //(SHR2(q[1], 1) ^ SHL2(q[1], 2) ^ ROL2(q[1], 13) ^ ROL2(q[1], 43)) +
		    //(SHR2(q[1 + 1], 2) ^ SHL2(q[1 + 1], 1) ^ ROL2(q[1 + 1], 19) ^ ROL2(q[1 + 1], 53)) +
		    //(SHR2(q[1 + 2], 2) ^ SHL2(q[1 + 2], 2) ^ ROL2(q[1 + 2], 28) ^ ROL2(q[1 + 2], 59)) +
		    //(SHR2(q[1 + 3], 1) ^ SHL2(q[1 + 3], 3) ^ ROL2(q[1 + 3], 4) ^ ROL2(q[1 + 3], 37)) +
		    //(SHR2(q[1 + 4], 1) ^ SHL2(q[1 + 4], 2) ^ ROL2(q[1 + 4], 13) ^ ROL2(q[1 + 4], 43)) +
		    //(SHR2(q[1 + 5], 2) ^ SHL2(q[1 + 5], 1) ^ ROL2(q[1 + 5], 19) ^ ROL2(q[1 + 5], 53)) +
		    (SHR2(q[1 + 6], 2) ^ SHL2(q[1 + 6], 2) ^ ROL2(q[1 + 6], 28) ^ ROL2(q[1 + 6], 59)) +
		    (SHR2(q[1 + 7], 1) ^ SHL2(q[1 + 7], 3) ^ ROL2(q[1 + 7], 4) ^ ROL2(q[1 + 7], 37)) +
		    (SHR2(q[1 + 8], 1) ^ SHL2(q[1 + 8], 2) ^ ROL2(q[1 + 8], 13) ^ ROL2(q[1 + 8], 43)) +
		    (SHR2(q[1 + 9], 2) ^ SHL2(q[1 + 9], 1) ^ ROL2(q[1 + 9], 19) ^ ROL2(q[1 + 9], 53)) +
		    (SHR2(q[1 + 10], 2) ^ SHL2(q[1 + 10], 2) ^ ROL2(q[1 + 10], 28) ^ ROL2(q[1 + 10], 59)) +
		    (SHR2(q[1 + 11], 1) ^ SHL2(q[1 + 11], 3) ^ ROL2(q[1 + 11], 4) ^ ROL2(q[1 + 11], 37)) +
		    (SHR2(q[1 + 12], 1) ^ SHL2(q[1 + 12], 2) ^ ROL2(q[1 + 12], 13) ^ ROL2(q[1 + 12], 43)) +
		    (SHR2(q[1 + 13], 2) ^ SHL2(q[1 + 13], 1) ^ ROL2(q[1 + 13], 19) ^ ROL2(q[1 + 13], 53)) +
		    (SHR2(q[1 + 14], 2) ^ SHL2(q[1 + 14], 2) ^ ROL2(q[1 + 14], 28) ^ ROL2(q[1 + 14], 59)) +
		    (SHR2(q[1 + 15], 1) ^ SHL2(q[1 + 15], 3) ^ ROL2(q[1 + 15], 4) ^ ROL2(q[1 + 15], 37)) +
		    ((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL2(msg[1], 1 + 1) +
		    ROL2(msg[1 + 3], 1 + 4)) ^ hash[1 + 7]);

	    q[2 + 16] = CONST_EXP2(2) +
		    ((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL2(msg[2], 2 + 1) +
		    ROL2(msg[2 + 3], 2 + 4) - ROL2(msg[2 + 10], 2 + 11)) ^ hash[2 + 7]);
	    q[3 + 16] = CONST_EXP2(3) +
		    ((make_uint2(0x5555554F, 0x65555555) + ROL2(msg[3], 3 + 1) +
		    ROL2(msg[3 + 3], 3 + 4) - ROL2(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	    q[4 + 16] = CONST_EXP2(4) +
		    ((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) +ROL2(msg[4], 4 + 1) +
		    ROL2(msg[4 + 3], 4 + 4) - ROL2(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	    q[5 + 16] = CONST_EXP2(5) +
		    ((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL2(msg[5], 5 + 1) +
		    ROL2(msg[5 + 3], 5 + 4) - ROL2(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);

	    #pragma unroll 3
	    for (int i = 6; i<9; i++) {
		    q[i + 16] = CONST_EXP2(i) +
			    ((vectorize((i + 16)*(0x0555555555555555ull)) + ROL2(msg[i], i + 1) -
			    ROL2(msg[i - 6], (i - 6) + 1)) ^ hash[i + 7]);
	    }

	    #pragma unroll 4
	    for (int i = 9; i<13; i++) {
		    q[i + 16] = CONST_EXP2(i) +
			    ((vectorize((i + 16)*(0x0555555555555555ull)) +
			    ROL2(msg[i + 3], i + 4) - ROL2(msg[i - 6], (i - 6) + 1)) ^ hash[i - 9]);
	    }

	    q[13 + 16] = CONST_EXP2(13) +
		    ((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL2(msg[13], 13 + 1) +
		    ROL2(msg[13 - 13], (13 - 13) + 1) - ROL2(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	    q[14 + 16] = CONST_EXP2(14) +
		    ((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL2(msg[14], 14 + 1) +
		    ROL2(msg[14 - 13], (14 - 13) + 1) - ROL2(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	    q[15 + 16] = CONST_EXP2(15) +
		    ((make_uint2(0x5555554B, 0xA5555555) + ROL2(msg[15], 15 + 1) +
		    ROL2(msg[15 - 13], (15 - 13) + 1) - ROL2(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);


	    uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	    uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

        uint2 h[16];

	    h[0] = (SHL2(XH64, 5) ^ SHR2(q[16], 5) ^ msg[0]) + (XL64 ^ q[24] ^ q[0]);
	    h[1] = (SHR2(XH64, 7) ^ SHL2(q[17], 8) ^ msg[1]) + (XL64 ^ q[25] ^ q[1]);
	    h[2] = (SHR2(XH64, 5) ^ SHL2(q[18], 5) ^ msg[2]) + (XL64 ^ q[26] ^ q[2]);
	    h[3] = (SHR2(XH64, 1) ^ SHL2(q[19], 5) ^ msg[3]) + (XL64 ^ q[27] ^ q[3]);
	    h[4] = (SHR2(XH64, 3) ^ q[20] ^ msg[4]) + (XL64 ^ q[28] ^ q[4]);
	    h[5] = (SHL2(XH64, 6) ^ SHR2(q[21], 6) ^ msg[5]) + (XL64 ^ q[29] ^ q[5]);
	    h[6] = (SHR2(XH64, 4) ^ SHL2(q[22], 6) ^ msg[6]) + (XL64 ^ q[30] ^ q[6]);
	    h[7] = (SHR2(XH64, 11) ^ SHL2(q[23], 2) ^ msg[7]) + (XL64 ^ q[31] ^ q[7]);

	    h[8] =  ROL2(h[4], 9)  + (XH64 ^ q[24] ^ msg[8]) + (SHL2(XL64, 8) ^ q[23] ^ q[8]);
	    h[9] =  ROL2(h[5], 10) + (XH64 ^ q[25]) + (SHR2(XL64, 6) ^ q[16] ^ q[9]);
	    h[10] = ROL2(h[6], 11) + (XH64 ^ q[26]) + (SHL2(XL64, 6) ^ q[17] ^ q[10]);
	    h[11] = ROL2(h[7], 12) + (XH64 ^ q[27]) + (SHL2(XL64, 4) ^ q[18] ^ q[11]);
	    h[12] = ROL2(h[0], 13) + (XH64 ^ q[28]) + (SHR2(XL64, 3) ^ q[19] ^ q[12]);
	    h[13] = ROL2(h[1], 14) + (XH64 ^ q[29]) + (SHR2(XL64, 4) ^ q[20] ^ q[13]);
	    h[14] = ROL2(h[2], 15) + (XH64 ^ q[30]) + (SHR2(XL64, 7) ^ q[21] ^ q[14]);
	    h[15] = ROL2(h[3], 16) + (XH64 ^ q[31] ^ msg[15]) + (SHR2(XL64, 2) ^ q[22] ^ q[15]);




		// Final
		#pragma unroll
		for(int i=0;i<16;i++)
		{
			msg[i].y = 0xaaaaaaaa;
			msg[i].x = 0xaaaaaaa0ul + (uint32_t)i;
		}
		bmw_compression512(h, msg);


        #pragma unroll
		for(int i=0;i<8;i++)
			pHash[i] = devectorize(msg[i+8]);
	}
}

__host__
void cuda_base_bmw512_cpu_hash_64(const uint32_t threads, uint32_t *d_hash) {

	const uint32_t threadsperblock = BMW512_TPB64;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_base_bmw512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

#pragma endregion

