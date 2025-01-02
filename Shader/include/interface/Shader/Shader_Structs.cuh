#ifndef SHADER_STRUCTS_CUH
#define SHADER_STRUCTS_CUH

#include <cuda_runtime.h>

__host__ __device__ struct Triangle {
    float3 pointA;
    float3 pointB;
    float3 pointC;

};

__host__ __device__ struct Cube {
    float test2;
};

__host__ __device__ struct Sphere {
    float3 position;
    float radiusSquared;
};

__host__ __device__ struct Matrix3x3 {
    float test3;
};

#endif