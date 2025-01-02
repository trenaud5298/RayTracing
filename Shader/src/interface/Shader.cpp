#include <Shader/Shader.h>
#include <Shader/Shader_Settings.h>
#include <Shader/Shader_Logging.h>
#include <Shader/Shader_World.h>
#include <Shader/Shader_ResourceManager.h>
#include <iostream>

namespace Shader::Logging {

}

namespace Shader {

    //Creates A Static Instance Of Shader_Settings That Can Only Be Accessed From This File

    bool Init(int resolutionX, int resolutionY, bool enableRuntimeLogging) {
        if(!Shader::Logging::InitLogging()) {
            Shader::Logging::log(LOG_TYPE_ERROR, "Failed To Initialized Logging Sub-Module");
            return false;
        }

        if(!Shader::ResourceManager::InitResourceManager()) {
            Shader::Logging::log(LOG_TYPE_ERROR, "Failed To Initialized Resource Manager Sub-Module");
            return false;
        }

        if(!Shader::Settings::InitSettings()) {
            Shader::Logging::log(LOG_TYPE_ERROR, "Failed To Initialized Settings Sub-Module");
            return false;
        }

        if(!Shader::World::InitWorld()) {
            Shader::Logging::log(LOG_TYPE_ERROR, "Failed To Initialized World Sub-Module");
            return false;
        }
        
        Shader::Logging::log(LOG_TYPE_INFO, "All Sub-Systems Initialized, Shader Initializaion Complete");
        return true; // Placeholder return
    }

    bool Shutdown() {
        // TODO: Implementation to be added later
        return false; // Placeholder return
    }

}
