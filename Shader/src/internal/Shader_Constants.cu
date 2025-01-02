#include <Shader/Shader_Constants.cuh>

// Define and initialize constant memory variables

__constant__ Triangle d_Triangles[256];

__constant__ Cube d_Cubes[256];

__constant__ Sphere d_Spheres[256];

__constant__ Matrix3x3 d_RotationMatrix;

__constant__ int d_testNum;
