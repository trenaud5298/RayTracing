#ifndef OBJECT3D_CUH
#define OBJECT3D_CUH

__host__ __device__ struct Sphere {
    float x;
    float y;
    float z;
    float r;
};

__host__ __device__ struct Cube {
    float lowX;
    float lowY;
    float lowZ;
    float highX;
    float highY;
    float highZ;
};



#endif