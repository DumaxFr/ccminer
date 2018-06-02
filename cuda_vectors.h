#include "cuda_helper.h"

/* Macros for uint2 operations (used by skein) */

__device__ __forceinline__
uint2 ROR8(const uint2 a) {
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x4321);
	result.y = __byte_perm(a.y, a.x, 0x4321);
	return result;
}

__device__ __forceinline__
uint2 ROL24(const uint2 a) {
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x0765);
	result.y = __byte_perm(a.y, a.x, 0x0765);
	return result;
}

static __device__ __forceinline__ uint2 operator+ (const uint2 a, const uint32_t b)
{
#if 0 && defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm(
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
	: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
#else
	return vectorize(devectorize(a) + b);
#endif
}

/* whirlpool ones */
#ifdef __CUDA_ARCH__
__device__ __forceinline__
uint2 ROL16(const uint2 a) {
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x1076);
	return result;
}
#else
#define ROL16(a) make_uint2(a.x, a.y) /* bad, just to define it */
#endif

static __forceinline__ __device__ __host__ uint4 operator+ (uint4 a, uint4 b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __forceinline__ __device__ __host__ uint4 operator^ (uint4 a, uint4 b) { return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ __host__ uint4 operator& (uint4 a, uint4 b) { return make_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w); }
static __forceinline__ __device__ __host__ uint4 operator>>(uint4 a, int b) { return make_uint4(a.x >> b, a.y >> b, a.z >> b, a.w >> b); }
static __forceinline__ __device__ __host__ uint4 operator<<(uint4 a, int b) { return make_uint4(a.x << b, a.y << b, a.z << b, a.w << b); }
static __forceinline__ __device__ __host__ uint4 operator* (uint4 a, int b) { return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __forceinline__ __device__ __host__  void operator^=(uint4 &a, uint4 b) { a = a ^ b; }
static __forceinline__ __device__ __host__  void operator+=(uint4 &a, uint4 b) { a = a + b; }



typedef struct __align__(32) uint8 {
	unsigned int s0, s1, s2, s3, s4, s5, s6, s7;
} uint8;

typedef struct __align__(64) uint16 {
	union {
		struct {unsigned int  s0, s1, s2, s3, s4, s5, s6, s7;};
		uint8 lo;
	};
	union {
		struct {unsigned int s8, s9, sa, sb, sc, sd, se, sf;};
		uint8 hi;
	};
} uint16;

static __inline__ __host__ __device__ uint16 make_uint16(
	unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3, unsigned int s4, unsigned int s5, unsigned int s6, unsigned int s7,
	unsigned int s8, unsigned int s9, unsigned int sa, unsigned int sb, unsigned int sc, unsigned int sd, unsigned int se, unsigned int sf)
{
	uint16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	t.s8 = s8; t.s9 = s9; t.sa = sa; t.sb = sb; t.sc = sc; t.sd = sd; t.se = se; t.sf = sf;
	return t;
}

static __forceinline__ __device__ __host__ uint16 operator^ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7,
		a.s8 ^ b.s8, a.s9 ^ b.s9, a.sa ^ b.sa, a.sb ^ b.sb, a.sc ^ b.sc, a.sd ^ b.sd, a.se ^ b.se, a.sf ^ b.sf);
}

static __forceinline__ __device__  __host__ uint16 operator+ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7,
		a.s8 + b.s8, a.s9 + b.s9, a.sa + b.sa, a.sb + b.sb, a.sc + b.sc, a.sd + b.sd, a.se + b.se, a.sf + b.sf);
}

static __forceinline__ __device__  __host__ uint16 operator- (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 - b.s0, a.s1 - b.s1, a.s2 - b.s2, a.s3 - b.s3, a.s4 - b.s4, a.s5 - b.s5, a.s6 - b.s6, a.s7 - b.s7,
		a.s8 - b.s8, a.s9 - b.s9, a.sa - b.sa, a.sb - b.sb, a.sc - b.sc, a.sd - b.sd, a.se - b.se, a.sf - b.sf);
}

static __forceinline__ __device__  __host__ void operator^= (uint16 &a, const uint16 &b) { a = a ^ b; }
static __forceinline__ __device__  __host__ void operator+= (uint16 &a, const uint16 &b) { a = a + b; }
static __forceinline__ __device__  __host__ void operator-= (uint16 &a, const uint16 &b) { a = a - b; }
