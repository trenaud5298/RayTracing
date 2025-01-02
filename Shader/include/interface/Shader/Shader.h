#ifndef SHADER_CUH
#define SHADER_CUH



/**
 * @file Shader.h
 * @brief Main Interface Namespace For The Shader Library, 
 *       Which Handles The Rendering Of A Scene Using A Ray Tracing Algorithm On The GPU.
 *
 * ================================================================================  
 * Shader Library Overview  
 * ================================================================================  
 * This Namespace Provides Core Functionality For Initializing, Shutting Down, Rendering  
 * The Scene, And Accessing The Rendered Image.  
 *  
 * Sub-Namespaces For Additional Functionality:  
 * 
 *  - `Shader::Settings`: Manages Runtime And Compile-Time Configurable Shader Settings. 
 *  
 *  - `Shader::Logging`: Provides Comprehensive Logging Capabilities For Debugging And Diagnostics. 
 *  
 *  - `Shader::World`: Handles Scene Editing And Management For Ray Tracing.  
 *  
 * ================================================================================  
 * Shader Library Requirements  
 * ================================================================================  
 * To Use The `Shader` Library, The Following Are Required:  
 *  
 *  - A Call To `init()` To Initialize The Library And Prepare The GPU For Rendering.  
 *  
 *  - A Call To `shutdown()` For Proper Memory Cleanup And Resource Deallocation.  
 *  
 * This Library Leverages CUDA For GPU Computation And Requires:  
 *  
 *  - An NVIDIA GPU With The Appropriate Drivers Installed.  
 *  
 *  - The `nvcc` Compiler For Compiling CUDA Code.  
 *  
 * It Is Designed For High-Performance Ray Tracing On The GPU, Leveraging Parallel Computation  
 * To Achieve Efficient Scene Rendering. The Only External Dependency Is An NVIDIA GPU And The CUDA  
 * Toolchain For Compilation.  
 *  
 * ================================================================================  
 * Important Notes  
 * ================================================================================  
 *  - Make Sure To Call `init()` Before Using Any Methods In This Namespace, And  
 *    `shutdown()` After All Rendering Tasks Are Completed To Avoid Memory Leaks.  
 * 
 *  - For Full Details On How Logging Works And Additional Configurations, Refer To The Official 
 *    Documentation At [insert location of official documentation here].
 *  
 * ================================================================================  
 * @author Tyler Renaud  
 * @date 12-17-2024  
 * @version 1.0 Alpha  
 */
namespace Shader {

//Note: Still Need To Add Documentation Location In Main Comment Above

    /**
     * @brief Shader Initialization Method. Required To Be Called Before Any Other Methods
     * Otherwise An Exception Will Be Thrown. 
     * 
     * This Method Is Expected To Be Paired With `shutdown()` To Ensure Proper Memory Cleanup.
     * `init()` Cannot Be Called A Second Time Until Matching `shutdown()` Method Is Called,
     * Otherwise Init Will Fail And Error Will Be Logged. Optional Paramaters Allow The
     * Starting Resolution To Be Set To Custom Input, Otherwise Default Of 1200x800 Will Be Used.
     * 
     * @param resolutionX Horizontal Resolution For Shader. Default Is `1200`.
     * 
     * @param resolutionY Vertical Resolution For Shader. Default Is `800`.
     * 
     * @param enableRuntimeLogging Determines If Runtime Logging Is Enabled For Shader. Default Is `false`.
     * Note: Logging Must Be Enabled At Compile Time For Logging To Be Possible. See Documentation For More Details
     * 
     * 
     * @return True If Initialization Is Succesful
     * 
     * @return Fales, If Initialization Fails. See `Shader_GetError()` For More Information
     */
    bool Init(int resolutionX = 1200, int resolutionY = 800, bool enableRuntimeLogging = false);

    /**
     * @brief Shader Shutdown Method. Required To Be Called After A Call To init() For 
     * Proper Memory Cleanup. 
     * 
     * In The Event That This Method Fails To Properly
     * Handle Shutdown, The Method Will Return False, Log An Error, And Also Have
     * The Shader Remain In An Initialized State, Requiring The Issue To Either Be
     * Fixed, And A New Call To shutdown() Or For The Program To Be Restarted 
     * Potentially Causing A Memory Leak In The Process.
     * 
     * @return True, If Shutdown Is Successful
     * @return False, If Shutdown Fails. See Shader_GetError() For More Information
     */
    bool Shutdown();


}

#endif