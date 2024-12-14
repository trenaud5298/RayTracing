#ifndef VECTORMATH_CUH
#define VECTORMATH_CUH

#include <cuda_runtime.h>

//Does Not Include cuda_runtime.h To Avoid Multiple Inclusions
//This File Handles Custom Float 3 Operators That Will Be Used To Simplify Code
//In The Kernel And Can Be Used Help Make Code More Readable When Dealing Heavily
//With float3 On CUDA Kernels

//NOTE: Prior To Including This File #include <cuda_runtime.h> Must Have Been Called
//NOTE: This File Can Only Be Included In A .cu File (Or A .cuh File That Is Only Included In .cu Files) In Order To Avoid Compilation Errors 


//Addition
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator+(const float3& a, const float& scalar) {
    return make_float3(a.x + scalar, a.y + scalar, a.z + scalar);
}

__device__ __forceinline__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ float3& operator+=(float3& a, const float& scalar) {
    a.x += scalar;
    a.y += scalar;
    a.z += scalar;
    return a;
}

//Subtraction
__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float& scalar) {
    return make_float3(a.x - scalar, a.y - scalar, a.z - scalar);
}

__device__ __forceinline__ float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__device__ __forceinline__ float3& operator-=(float3& a, const float& scalar) {
    a.x -= scalar;
    a.y -= scalar;
    a.z -= scalar;
    return a;
}

//Multiplication
__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float& scalar) {
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__device__ __forceinline__ float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

__device__ __forceinline__ float3& operator*=(float3& a, const float& scalar) {
    a.x *= scalar;
    a.y *= scalar;
    a.z *= scalar;
    return a;
}

//Division
__device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, const float& scalar) {
    return make_float3(a.x / scalar, a.y / scalar, a.z / scalar);
}

__device__ __forceinline__ float3& operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

__device__ __forceinline__ float3& operator/=(float3& a, const float& scalar) {
    a.x /= scalar;
    a.y /= scalar;
    a.z /= scalar;
    return a;
}

//Additional Vector Operations

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    //Uses Quick (a*b)+c operator
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float length(const float3& a) {
    return sqrtf(fmaf(a.x, a.x, fmaf(a.y, a.y, a.z * a.z)));
}

__device__ __forceinline__ float inverseLength(const float3& a) {
    return rsqrtf(fmaf(a.x, a.x, fmaf(a.y, a.y, a.z * a.z)));
}

__device__ __forceinline__ float3 normalize(float3& a) {
    float inverseLength = rsqrtf(fmaf(a.x, a.x, fmaf(a.y, a.y, a.z * a.z)));
    a.x *= inverseLength;
    a.y *= inverseLength;
    a.z *= inverseLength;
    return a;
}


#endif