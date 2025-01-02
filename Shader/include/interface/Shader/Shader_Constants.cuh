#ifndef SHADER_CONSTANTS_CUH
#define SHADER_CONSTANTS_CUH

#include <cuda_runtime.h>
#include <Shader/Shader_Structs.cuh>

//Constant Memory Storage For 
extern  __constant__  Triangle   d_Triangles[256];

extern  __constant__  Cube       d_Cubes[256];

extern  __constant__  Sphere     d_Spheres[256];

extern  __constant__  Matrix3x3  d_RotationMatrix;

extern __constant__ int d_testNum;



#endif