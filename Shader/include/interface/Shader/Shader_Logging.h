#ifndef SHADER_LOGGING_H
#define SHADER_LOGGING_H

#include <iostream>

/**
 * @file Shader_Logging.h
 * @brief Namespace For Handling Logging And Diagnostics In The Shader Library.
 *
 * ================================================================================  
 * Logging Overview  
 * ================================================================================  
 * The `Shader::Logging` Namespace Provides Tools To Log Information, Warnings, And Errors.  
 * Logging Is Controlled Via Compile-Time Options:  
 *  
 *  - Compile-Time: Define `ENABLE_LOGGING` To Include Logging Features.  
 *        
 * Available Logging Methods Include:  
 *  - Retrieving Logs Of Specific Types (`Error`, `Info`, `Warning`).  
 * 
 *  - Checking The Current Logging Status.  
 * 
 *  - Adjusting Logging Behavior Dynamically During Runtime.  
 *  
 * ================================================================================  
 * Logging Best Practices  
 * ================================================================================  
 *  - Enable And Use Logging In Debug Builds And Disable In Release Builds (Default Settings For This Library).  
 *  
 *  - Use `LOG(type, message)` Macro To Avoid Calls To `log()` When Logging Is Disabled  
 * ================================================================================  
 * @note Refer To `Shader_Settings.h` For Adjusting Logging Preferences At Runtime.  
 */
namespace Shader::Logging {
    //Defines The Different Types Of Logs Using Binary Allowing For A LogMessage To Be Of Multiple Log Types
    #define LOG_TYPE_ERROR    0x01
    #define LOG_TYPE_WARNING  0x02
    #define LOG_TYPE_INFO     0x04
    #define LOG_TYPE_DEBUG    0x08
    #define LOG_TYPE_ALL (LOG_TYPE_ERROR | LOG_TYPE_WARNING | LOG_TYPE_INFO | LOG_TYPE_DEBUG)



    /**
     * This Method Handles The Initialization Of Logging For The Shader Library.
     * 
     * This Method Is Called Automatically From `Shader::init()`. If Logging Is 
     * Enabled At Compile Time It Will Create A Buffer To Store Log Messages With
     * Size Equal To `MAX_LOG_LENGTH`.
     * 
     * Logging Is Controlled By The Compile Time Macro `ENABLE_LOGGING`.
     * See Full Documentation For Details On How To Manage Logging.
     * 
     * @return True, If Logging Is Initialized Successfully.
     * @return False, If Logging Initialization Fails. Will Print Out Additional Information In This Case
     */
    bool InitLogging(); 

    /**
     * This Method Handles The Shutdown Of The Logging Sub-Module
     * 
     * This Method Is Called Automatically From `Shader::Shutdown` And Frees
     * Any Allocated Memory And Properly Cleans Up All Shader Log Files.
     * 
     * @return True, If Logging Is Sucessfully Shutdown
     * @return False, If Logging Shutdown Fails
     */
    bool ShutdownLogging();

    /**
     * @brief This Method Handles Forcibly Logging A Message 
     * Reguardeless Of Compile Time Instructions
     * 
     * This Message Will Either Add Specified Message To Log
     * Or Print Out Log Message To Standard Output Based On
     * Whether Logging Is Enabled Or Not. 
     * 
     * Intended To Only Be Called By Critical Internal Systems,
     * Or For Serious Errors. Otherwise, The Use Of log() Should
     * Instead Be Used To Reduce Wasted Time Spent Logging When 
     * Logging Is Disabled
     */
    void FORCE_LOG(unsigned char logType, const std::string& logMessage);

    /**
     * This Method Handles Adding A Log Message To The Log Buffer.
     * 
     * This Method Calls FORCE_LOG When Logging Is Enabled, Otherwise
     * The Compiler Should Optimize This inline Function Away Due
     * To The Blank Definition.
     * 
     * If You Wish To Log A Message Reguardless Of Logging Status You
     * Can Directly Call To FORCE_LOG Which Will Either Log Your Message
     * Normaly Or Simply Print Out Your Log Message Depending On Whether
     * Logging Is Enabled Or Not
     */
    void log(unsigned char logType, const std::string& logMessage);


    /**
     * @brief This Method Returns A Boolean Value Representing Whether Logging Is Currently Enabled Or Not.
     * 
     * Note: This Method Only Returns The Current Status Of Whether Logging Is Actively Working, And Therefor
     * If Logging Has Not Been Properly Initialized, Then Method Will Also Return False
     * 
     * @return True, If Logging Is Enabled
     * @return False, If Logging Is Disabled Or If Logging Has Not Been Initialized
     */
    bool IsLoggingEnabled();

    /**
     * @brief Retrieves the last log message of type "ERROR".
     * 
     * This method will return the most recent error message logged. If no error has been logged, the
     * method will return an empty string.
     * 
     * @return An std::string containing the last error log message, or null if no error has been logged.
     */
    std::string GetLastErrorLogMessage();

    /**
     * @brief Retrieves the entire error log.
     * 
     * This method will return all the error messages logged, separated by new line characters in the order
     * they were added. If no error messages exist, the method will return an empty string.
     * 
     * @return An std::string containing all error messages, separated by new lines.
     */
    std::string GetErrorLog();

    /**
     * @brief Retrieves the last log message of type "WARNING".
     * 
     * This method will return the most recent warning message logged. If no warning has been logged, the
     * method will return an empty string.
     * 
     * @return An std::string containing the last warning log message, or null if no warning has been logged.
     */
    std::string GetLastWarningLogMessage();

    /**
     * @brief Retrieves the entire warning log.
     * 
     * This method will return all the warning messages logged, separated by new line characters in the order
     * they were added. If no warning messages exist, the method will return an empty string.
     * 
     * @return An std::string containing all warning messages, separated by new lines.
     */
    std::string GetWarningLog();
    
    /**
     * @brief Retrieves the last log message of type "INFO".
     * 
     * This method will return the most recent information log message. If no info message has been logged, the
     * method will return an empty string.
     * 
     * @return An std::string containing the last info log message, or null if no info has been logged.
     */
    std::string GetLastInfoLogMessage();

    /**
     * @brief Retrieves the entire info log.
     * 
     * This method will return all the info messages logged, separated by new line characters in the order
     * they were added. If no info messages exist, the method will return an empty string.
     * 
     * @return An std::string containing all info messages, separated by new lines.
     */

    std::string GetInfoLog();
    /**
     * @brief Retrieves the last log message of type "DEBUG".
     * 
     * This method will return the most recent debug message logged. If no debug has been logged, the
     * method will return an empty string.
     * 
     * @return An std::string containing the last debug log message, or null if no debug has been logged.
     */
    std::string GetLastDebugLogMessage();

    /**
     * @brief Retrieves the entire debug log.
     * 
     * This method will return all the debug messages logged, separated by new line characters in the order
     * they were added. If no debug messages exist, the method will return an empty string.
     * 
     * @return An std::string containing all debug messages, separated by new lines.
     */
    std::string GetDebugLog();

    /**
     * @brief Retrieves the most recent log message, regardless of type.
     * 
     * This method will return the most recent log message from any type (ERROR, INFO, WARNING, DEBUG). If no logs
     * have been recorded, the method will return an empty string.
     * 
     * @return An std::string containing the most recent log message, or null if no logs have been recorded.
     */
    std::string GetLastLogMessage();

    /**
     * @brief Retrieves the entire log, with each message separated by a new line.
     * 
     * This method will return all the log messages recorded, regardless of type, separated by new line
     * characters in the order they were added. If no log messages exist, the method will return an empty string.
     * 
     * @return An std::string containing all log messages, separated by new lines.
     */
    std::string GetLog();

}; //Namespace Shader::Logging




#endif