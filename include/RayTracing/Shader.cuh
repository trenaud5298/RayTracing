#ifndef RAYTRACING_SHADER_CUH
#define RAYTRACING_SHADER_CUH

#include <cuda_runtime.h> //Also Include <cuda_runtime.h> Through This Include
#include <curand_kernel.h>
#include <iostream>

/**
 * Static Class That Ensures No Instances Of A Shader Class Are Ever Instantiated
 * 
 * 
 */
class Shader {

//Methods
public:

    /**
     * Calls Primary Shader Initialization Method With resolutionX = 1200 And
     * resolutionY = 800
     */
    static void init();

    /**
     * Primary Initialization Method For Shader
     * -Required To Be Called Before Majority Of Methods
     * -Handle Any Default Initialization Required For Shader Class To Run
     * 
     * @param resolutionX Horiozntal Resolution For Rendered Image
     * @param resolutionY Vertical Resolution For Rendered Image
     * 
     * @throws std::runtime_error() If Shader Has Already Been Initialized
     */
    static void init(int resolutionX, int resolutionY);

    /**
     * Sets Shader's Resolution To Passed In Arguments
     * -Updates Both Host And Device Memory Related To Resolution
     * -Resets Any Stored Image Data During Process
     * 
     * @param resolutionX New Horizontal Resolution For Shader
     * @param resolutionY New Vertical Resolution For Shader
     */
    static void setResolution(int resolutionX, int resolutionY);

    /**
     * Calls The Following runKernel() Method With numberOfPasses=1
     */
    static void runKernel();

    /**
     * Runs Primary CUDA Kernel Responsible For Updating Image numberOfPasses Times
     * -
     * 
     */
    static void runKernel(size_t numberOfPasses);

    /**
     * This Method Copies The Raw Image Data From Device Memory
     * To Host Memory And Then Returns A Pointer To Copied Memory
     * 
     * Note: Although It Is Possible To Make Changes To Returned Image
     * Memory, These Changes Will Not Be Reflected On GPU Memory And
     * Will Therefore Be Written Over Next Time This Method Is Called.
     * 
     * This Memory Will Also Be Wiped Upon A Call To setResolution(). 
     */
    static float* getRawImageData();

    /**
     * This Method Copies The Raw Image Data From Device Memory
     * To Host Memory And Then Saves It As A .hdr File
     * 
     * Note: Every Call To This Method Results In A Memory Copy From
     * Device To Host Even If No Update Has Been Made On Device.
     * This Method Should Not Be Called Repeatedly If Performace
     * Is A Concern And Should Instead Call getRawImageData() Once
     * To Get A Pointer To Raw Data And Then Save It Repeatedly If Needed
     */
    static void saveImage(const std::string& filePath);


    /**
     * Need To Add Methods That Handle Enviorment Updates And One For Camera Updates As Well(Useful Since Camera Moves A Lot)
     */
    

    static void updateWorldData(/**Should Take In A Full World Class That WIll Update Everything */);

    //These following methods are just ideas for now and may need to be updated/changed
    static void updateBoundingBoxes();

    static void updateSpheres();

    static void updateTriangles();

    static void updateCamera();


private:
    /**
     * Deleted Shader Constructor To Ensure Static Class
     */
    Shader() = delete;
    /**
     * Deleted Shader Destructor To Ensure Static Class
     */
    ~Shader() = delete;
    /**
     * Deleted Shader Copying To Ensure Static Class
     */
    Shader(const Shader&) = delete;
    /**
     * Deleted Shader Assignment To Ensure Static Class
     */
    Shader& operator=(const Shader&) = delete;

    /**
     * This Private Method Is Called Upon Shader Initialization
     * And Loads The Default Data For Shader Class. This Method
     * Will Also Hold All Of The Basic World Data Within Itself
     * And Therefor Relies On No Other Data Files For Default 
     * Enviorment Data.
     */
    static void loadDefaultData();

    /**
     * This Method Handles Allocation Of All Dynamic Memory That
     * Partains To Storing Image Data. This Method Handles Both Host
     * And Device Memory And Should Only Be Called Upon Shader
     * Initialization Or Upon A Resolution Change
     */
    static void allocateDynamicImageMemory();

    /**
     * This Method Frees All Allocated Dynamic Memory That Partains
     * To Storing Image Data. This Method Handles Both Host And Device
     * Memory And Should Only Be Called Upon Shader Clean Up Or
     * Upon A Resolution Change
     */
    static void freeDynamicImageMemory();

    /**
     * This Method Handles Allocation Of All Dynamic Memory That
     * Partains To Storing World Data. This Method Handles Both Host
     * And Device Memory And Should Only Be Called Upon Shader
     * Initialization Or Upon A Complete World Data Wipe Via The
     * freeDynamicWorldMemory() Method Upon A Complete World Change
     */
    static void allocatedDynamicWorldMemory(); //Need Explanation
    
    /**
     * This Method Frees All Allocated Dynamic Memory That Partains
     * To Storing World Data. This Method Handles Both Host And Device
     * Memory And Should Only Be Called Upon Shader Clean Up Or If
     * All World Data Must Be Changed Or Reset In Its Entirety
     */
    static void freeDynamicWorldMemory();

    /**
     * This Method Is Called At The Start Of Every Publically Accesible Functions (Except init())
     * And Ensures That Shader Has Been Initialized. If Shader Has Not Been Initialized
     * Then An Exception Will Be Thrown.
     * @throws std::runtime_error() - If Shader Has Not Been Initialized
     */
    static inline void ensureInit() {
        if(!Shader::isInitialized) {
            throw std::runtime_error("Shader Must Be Initialized Before Use");
        }
    }

    /**
     * This Method Is Called Whenever An Update To Shader's Resolution Is Initiated
     * In Order To Ensure That The New Resolution Is Within The Requirements
     * For The Shader. Shader Resolution Must Be Between 200 And 32000
     * @param resolutionX Horizontal Resolution To Be Checked
     * @param resolutionY Vertical Resolution To Be Checked
     * @throws std::runtime_error() - If Input Resolution Is Not Valid
     */
    static inline void ensureValidResolution(int resolutionX, int resolutionY) {
        if(resolutionX > 32000) { throw std::runtime_error("Invalid Horizontal Resolution! Shader Resolution Must Be Less Than 32,000! Attempted Horizontal Resolution: " + resolutionX); }
        if(resolutionY > 32000) { throw std::runtime_error("Invalid Vertical Resolution! Shader Resolution Must Be Less Than 32,000! Attempted Vertical Resolution: " + resolutionX); }
        if(resolutionX < 200) { throw std::runtime_error("Invalid Horizontal Resolution! Shader Resolution Must Be Greater Than 200! Attempted Horizontal Resolution: " + resolutionX); }
        if(resolutionY < 200) { throw std::runtime_error("Invalid Vertical Resolution! Shader Resolution Must Be Greater Than 200! Attempted Vertical Resolution: " + resolutionX);}
    }



//Data (Default Values For Static Varialbes Is In Comments)
public:
    //IDK what to put here for now, or if I even have a need for any globally accessible values for the shader

private:
//Shader State Data

    static bool isInitialized;                           //Default Value Is False;



//Shader Image Data

    //Image Resolution Data
    static int resolutionX;                              //Default Value Is 1200
    static int resolutionY;                              //Default Value Is 800
    static int currentFrame;                             //Default Value Is 1
    //Host Image Memory
    static float* rawImageData;                          //Default Value Is nullptr
    //Device Image Memory
    static float* device_rawImageData;                   //Default Value Is nullptr

};

#endif