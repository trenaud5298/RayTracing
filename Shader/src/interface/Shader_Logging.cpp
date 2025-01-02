#include <Shader/Shader_Logging.h>
#include <Shader/Shader_HostMemory.h>
#include <iostream>

namespace Shader::Logging {

#ifdef ENABLE_LOGGING
#pragma region Logging Enabled
//This Section Of Code Is Only Used When ENABLE_LOGGING Is Defined At Compile Time

    bool InitLogging() {
        if(Shader::HostMemory::loggingInitialized) {
            FORCE_LOG(LOG_TYPE_ERROR, "Logging Sub-Module Cannot Be Initialized Twice");
            return false;
        }
        //Creates Shader Log
        Shader::HostMemory::shaderLog = new ShaderLog();
        //Checks For Proper Creation Of ShaderLog Object
        if(!Shader::HostMemory::shaderLog) {
            FORCE_LOG(LOG_TYPE_ERROR, "Failed To Create Shader Log Object During Logging Sub-Module Initialization");
            return false;
        }
        //All Checks Have Passed At This Point So Set Logging Initialization Variable To True
        Shader::HostMemory::loggingInitialized = true;
        log(LOG_TYPE_INFO, "Logging Sub-Module Initialized");
        return true;
    }

    bool ShutdownLogging() {
        if(!Shader::HostMemory::loggingInitialized) {
            return false; //Can Not Shut Down Logging Sub-Module If It Is Not Initialized
        }

        if(Shader::HostMemory::shaderLog) {
            delete Shader::HostMemory::shaderLog;
        }

        //Ensure That It Has Been Properly Deleted
        if(Shader::HostMemory::shaderLog) {
            return false;
        }

        Shader::HostMemory::loggingInitialized = false;
        return true;
    }

    void FORCE_LOG(unsigned char logType, const std::string& logMessage) {
        if(!Shader::HostMemory::loggingInitialized) {
            std::cout<< LogEntry(LOG_TYPE_WARNING, "Logging Sub-Module Has Not Been Initialized! Attempting To Forcefully Log Message To Standard Output!").toString() <<"\n";
            std::cout<< LogEntry(logType, logMessage).toString() <<"\n\n";
            return;
        }
        Shader::HostMemory::shaderLog->addEntry(logType, logMessage);
    }

    void log(unsigned char logType, const std::string& logMessage) {
        FORCE_LOG(logType, logMessage);
    }

    bool IsLoggingEnabled() {
        return Shader::HostMemory::loggingInitialized;
    }

    std::string GetLastErrorLogMessage() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLastEntryOfTypeAsString(LOG_TYPE_ERROR);
    }

    std::string GetErrorLog() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLogOfTypeAsString(LOG_TYPE_ERROR);
    }

    std::string GetLastWarningLogMessage() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLastEntryOfTypeAsString(LOG_TYPE_WARNING);
    }

    std::string GetWarningLog() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLogOfTypeAsString(LOG_TYPE_WARNING);
    }

    std::string GetLastInfoLogMessage() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLastEntryOfTypeAsString(LOG_TYPE_INFO);
    }

    std::string GetInfoLog() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLogOfTypeAsString(LOG_TYPE_INFO);
    }

    std::string GetLastDebugLogMessage() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLastEntryOfTypeAsString(LOG_TYPE_DEBUG);
    }

    std::string GetDebugLog() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLogOfTypeAsString(LOG_TYPE_DEBUG);
    }

    std::string GetLastLogMessage() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLastEntryOfTypeAsString(LOG_TYPE_ALL);
    }

    std::string GetLog() {
        if(!Shader::HostMemory::loggingInitialized) {
            return "[ERROR] Logging Sub-Module Is Not Initialized";
        }
        // Default implementation
        return Shader::HostMemory::shaderLog->getLogOfTypeAsString(LOG_TYPE_ALL);
    }

#pragma endregion
#else

#pragma region test
//This Section Of Code Is Used When ENABLE_LOGGING Is Not Defined At Compile Time

    bool InitLogging() {
        if(Shader::HostMemory::loggingInitialized) {
            FORCE_LOG(LOG_TYPE_ERROR, "Logging Sub-Module Cannot Be Initialized Twice");
            return false;
        }
        Shader::HostMemory::loggingInitialized = true;
        FORCE_LOG(LOG_TYPE_WARNING, "Logging Has Been Disabled At Compile Time");
        return true;
    }

    void FORCE_LOG(unsigned char logType, const std::string& logMessage) {
        if(!Shader::HostMemory::loggingInitialized) {
            std::cout<< LogEntry(LOG_TYPE_WARNING, "Logging Sub-Module Has Not Been Initialized! Attempting To Forcefully Log Message To Standard Output!").toString() <<"\n";
        }
        std::cout<< LogEntry(logType,logMessage).toString() <<"\n";
    }

    void log(unsigned char logType, const std::string& logMessage) {
        //Do Nothing (Hopefully Optimized Out By Compiler, But Either Way It Will Drastically Reduce Any Performance Impact Caused By Logging)
    }

    bool IsLoggingEnabled() {
        return false;
    }

    std::string GetLastErrorLogMessage() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetErrorLog() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetLastWarningLogMessage() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetWarningLog() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetLastInfoLogMessage() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetInfoLog() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetLastDebugLogMessage() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetDebugLog() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetLastLogMessage() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

    std::string GetLog() {
        return "[ERROR] Logging Is Disabled At Compile Time";
    }

#pragma endregion
#endif

};
