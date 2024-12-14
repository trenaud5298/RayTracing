#include <RayTracing/Shader.cuh>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RayTracing/stb_image_write.h>
#include <RayTracing/VectorMath.cuh>

#pragma region Static Variables
//Default Shader Initialization State (Changes Upon Successful Call To Shader::init())
bool Shader::isInitialized = false;

//Image Data For Shader
    //Host Image Data
int Shader::resolutionX = 1200;
int Shader::resolutionY = 800;
int Shader::currentFrame = 1;
float* Shader::rawImageData = nullptr;
    //Device Image Data
__constant__ int DEVICE_RESOLUTION_X;
__constant__ int DEVICE_RESOLUTION_Y;
__constant__ curandState* DEVICE_RANDOM_GENERATORS;
float* Shader::device_rawImageData = nullptr;

//Ray Tracing Device Config Data



#pragma endregion


#pragma region Device Code

//Additional Vector Methods That Require Compilation



__global__ void initializeDynamicImageMemory(float* rawImageData, unsigned long long randomSeed) {
    //X and Y Corrdinates Corresponding To Image Coordinates
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x > DEVICE_RESOLUTION_X || y > DEVICE_RESOLUTION_Y) {
        return;
    }
    int threadID = y * DEVICE_RESOLUTION_X + x;
    //Initialize Corresponding CurandState (Random Number Generator)
    curand_init(randomSeed,threadID,0,&DEVICE_RANDOM_GENERATORS[threadID]);
    //Set Initial Pxiel Values For Raw Data
    rawImageData[threadID*3 + 0] = curand_uniform(&DEVICE_RANDOM_GENERATORS[threadID]);
    rawImageData[threadID*3 + 1] = curand_uniform(&DEVICE_RANDOM_GENERATORS[threadID]);
    rawImageData[threadID*3 + 2] = curand_uniform(&DEVICE_RANDOM_GENERATORS[threadID]);
}


__device__ void initialRay() {

}


__global__ void updateImage(int currentFrame, float* rawImageData) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x > DEVICE_RESOLUTION_X || y > DEVICE_RESOLUTION_Y) {
        return;
    }

    int threadID = y * DEVICE_RESOLUTION_X + x;
    curandState* randomGenerator = &DEVICE_RANDOM_GENERATORS[threadID];
    float3 a = {1.0f,2.0f,3.0f};
    float3 b = {3.0f,2.0f,1.0f};
    float3 c = cross(a,b);
    c = {curand_normal(randomGenerator),curand_normal(randomGenerator),curand_normal(randomGenerator)};


    //Randomize Pixel Values
    float weightInTotalImage = 1.0f/currentFrame;
    rawImageData[threadID*3 + 0] = ( rawImageData[threadID*3 + 0] * (1-weightInTotalImage) ) + ( c.x * weightInTotalImage );
    rawImageData[threadID*3 + 1] = ( rawImageData[threadID*3 + 1] * (1-weightInTotalImage) ) + ( c.y * weightInTotalImage );
    rawImageData[threadID*3 + 2] = ( rawImageData[threadID*3 + 2] * (1-weightInTotalImage) ) + ( c.z * weightInTotalImage );
}

#pragma endregion




#pragma region Host Code


void Shader::init() {
    //Calls Shader With Default Parameters
    Shader::init(1200,800);
}


void Shader::init(int resolutionX, int resolutionY) {
    //Throws Exception If Shader
    if(Shader::isInitialized) {
        throw std::runtime_error("Shader Cannot Be Initialized Twice");
    }

//Note:Does Not Call setResolution() Since Shader Has Not Been Initialized And Avoids Additional Call To freeDynamicImageMemory() When Not Needed
//Begin Image Memory Allocation
    //Ensures Input Resolution Is Valid
    Shader::ensureValidResolution(resolutionX,resolutionY);
    //Updates Internal Shader Resolution
    Shader::resolutionX = resolutionX; 
    Shader::resolutionY = resolutionY;
    //Sets Current Frame To 1
    Shader::currentFrame = 1;

    //Allocates Dynamic Image Memory
    Shader::allocateDynamicImageMemory();
//Ends Image Memory Allocation


}


void Shader::setResolution(int resolutionX, int resolutionY) {
    //Ensures Shader Has Been Initialized Prior To Call
    Shader::ensureInit();

    //Ensures Valid Resolution Input (Splits Up If Statements For Clearer Error Messages)
    Shader::ensureValidResolution(resolutionX,resolutionY);

    //Free Previous Image Data
    Shader::freeDynamicImageMemory();

    //Updates Internal Shader Resolution
    Shader::resolutionX = resolutionX; 
    Shader::resolutionY = resolutionY;
    Shader::currentFrame = 1;

    //Update Dynamic Memory With New Image Data
    Shader::allocateDynamicImageMemory();
}


void Shader::allocateDynamicImageMemory() {
    //Get Total Pixel Count
    int totalPixelCount = Shader::resolutionX * Shader::resolutionY;

    //Handles Host Dynamic Memory
    Shader::rawImageData = new float[ totalPixelCount * 3 ];

    //Handles Device Dynamic Memory
        //Resolution Data
    cudaMemcpyToSymbol(DEVICE_RESOLUTION_X, &Shader::resolutionX, sizeof(int));
    cudaMemcpyToSymbol(DEVICE_RESOLUTION_Y, &Shader::resolutionY, sizeof(int));
        //Raw Image data
    cudaMalloc(&Shader::device_rawImageData, totalPixelCount * 3 * sizeof(float));
        //Curand States (Used For Random Number Generation For Each Thread)
    curandState* device_randomGenerators = nullptr;
    cudaMalloc(&device_randomGenerators, totalPixelCount * sizeof(curandState));
    cudaMemcpyToSymbol(DEVICE_RANDOM_GENERATORS, &device_randomGenerators, sizeof(curandState*));

    //Handles CurandState Initialization And Addition Device Memory Initialization
        //Get Dimensions For CUDA Kernel
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(
        ceil(static_cast<float>(Shader::resolutionX)/threadsPerBlock.x), 
        ceil(static_cast<float>(Shader::resolutionY)/threadsPerBlock.y)
    );
        //Get Seed For Random Generator Initialization
    unsigned long long randomSeed = static_cast<unsigned long long>(std::time(nullptr));
        //Call Kernel
    initializeDynamicImageMemory<<<blocksPerGrid,threadsPerBlock>>>(Shader::device_rawImageData,randomSeed);
        //Handle Any Potential Errors
    cudaError_t initializationError = cudaDeviceSynchronize();
    if(initializationError) {
        std::cerr<<"CUDA Initialization Dynamic Image Memory Error: "<<initializationError<<std::endl;
    }
    std::cout<<"Everything Initialized"<<std::endl;
}


void Shader::freeDynamicImageMemory() {
    if(Shader::rawImageData) {
        delete[] Shader::rawImageData;
        Shader::rawImageData = nullptr;
    }

    if(Shader::device_rawImageData) {
        cudaFree(Shader::device_rawImageData);
        Shader::device_rawImageData = nullptr;
    }
}


void Shader::runKernel() {
    Shader::runKernel(1);
}

void Shader::runKernel(size_t numberOfPasses) {
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(
        ceil(static_cast<float>(Shader::resolutionX)/threadsPerBlock.x), 
        ceil(static_cast<float>(Shader::resolutionY)/threadsPerBlock.y)
    );
    for(size_t i = 0; i < numberOfPasses; i ++) {
        updateImage<<<blocksPerGrid,threadsPerBlock>>>(Shader::currentFrame, Shader::device_rawImageData);
        cudaDeviceSynchronize();
        Shader::currentFrame++;
    }
}


float* Shader::getRawImageData() {
    cudaMemcpy(Shader::rawImageData,Shader::device_rawImageData,Shader::resolutionX*Shader::resolutionY*3*sizeof(float),cudaMemcpyDeviceToHost);
    return Shader::rawImageData;
}

void Shader::saveImage(const std::string& filePath) {
    cudaMemcpy(Shader::rawImageData,Shader::device_rawImageData,Shader::resolutionX*Shader::resolutionY*3*sizeof(float),cudaMemcpyDeviceToHost);

    // Save the HDR image using raw float data
    if (!stbi_write_hdr(filePath.c_str(), Shader::resolutionX, Shader::resolutionY, 3, Shader::rawImageData)) {
        throw std::runtime_error("Failed to save HDR image: " + filePath);
    }

    std::cout << "HDR image saved successfully to: " << filePath << std::endl;
}




#pragma endregion