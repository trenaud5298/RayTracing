#ifndef SHADER_SETTINGS_H
#define SHADER_SETTINGS_H


/**
 * @file Shader_Settings.h
 * @brief Namespace For Configuring And Managing Shader Settings.
 *
 * ================================================================================  
 * Shader Settings Overview  
 * ================================================================================  
 * The `Shader::Settings` Namespace Provides Methods And Utilities To Configure Runtime  
 * And Compile-Time Settings For The Shader Library, Including:  
 *  
 *  - Resolution Of The Rendered Image.  
 * 
 *  - Number Of Rays Per Pixel Used In The Ray Tracing Algorithm. 
 *  
 *  - Other Configurable Options For Performance And Quality.  
 *  
 * ================================================================================  
 * Important Notes  
 * ================================================================================  
 *  - Settings Configured At Runtime Will Take Effect Immediately, Unless Specified Otherwise.  
 * 
 *  - Use `Shader::Settings` Methods To Dynamically Adjust Parameters Without Reinitializing The Library.  
 * 
 *  - See `Shader::Logging` For Methods Related To Changing Runtime Logging Settings 
 */
namespace Shader::Settings {

    /**
     * This Method Handles The Initialization Of Settings For The Shader Library. 
     * 
     * This Method Should Is Called Automatically From `Shader::init()` 
     * And Handles The Default Assignment Of Shader Settings. This Method
     * Assigns All Of The Settings Variables To Their Default Value. These
     * Values Can Be Modified At Compile Time Through Macros. See Full
     * Documentation For More Details
     * 
     * @return True, If Settings Are Configured And Set Properly
     * @return False, If Settings Are Unable To Be Configured. See Error Logs For More Details.
     */
    bool InitSettings();

    /**
     * TODO: need to add <<<<<<<<<<<===========================================
     */
    bool ShutdownSettings();

    /**
     * This Method Attemps To Change The Shader Resolution.
     * 
     * This Method Attempts To Change The Shader Resolution Value To
     * Whatever Values Are Passed In. Resolution Values Must Be Within
     * The Range `[200, 32000]` And If Any Values Are Not In This Range, The
     * Resolution Will Remain Unchanged And An Error Log Will Be Generated.
     * 
     * If The Passed In Value For One Of The Resolution Value Is `0`, An Info
     * Log Will Instead Be Generated Stating That The Desired Resolution Parameter
     * Shall Remain Unaffected As It Assumes This Behavior To Be Intentioanl
     */
    bool SetResolution(int resolutionX = 0, int resolutionY = 0);


}


#endif