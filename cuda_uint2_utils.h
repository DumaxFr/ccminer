#ifndef CUDA_UINT2_UTILS
#define CUDA_UINT2_UTILS

#include <vector_functions.h>


#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint2 cuda_swab64_U2(uint2 a)
{
	// Input:       77665544 33221100
	// Output:      00112233 44556677
	uint2 result;

	result.y = __byte_perm(a.x, 0, 0x0123);
	result.x = __byte_perm(a.y, 0, 0x0123);

	return result;
}
#else
// fake host version
#define cuda_swab64_U2(x) (x)
#endif // __CUDA_ARCH__


// perform asm xor when available
__device__ __forceinline__ uint2 xor3x(const uint2 a,const uint2 b,const uint2 c){
	uint2 result;
	#if __CUDA_ARCH__ >= 500
		asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.x) : "r"(a.x), "r"(b.x),"r"(c.x)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
		asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.y) : "r"(a.y), "r"(b.y),"r"(c.y)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	#else
		result.x = a.x ^ b.x ^ c.x;
		result.y = a.y ^ b.y ^ c.y;
	#endif
	return result;
}


#endif // !CUDA_UINT2_UTILS


