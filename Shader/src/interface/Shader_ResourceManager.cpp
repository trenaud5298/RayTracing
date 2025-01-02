#include <Shader/Shader_ResourceManager.h>
#include <Shader/Shader_HostMemory.h>
#include <Shader/Shader_FileCatalog.h>
#include <Shader/Shader_Logging.h>

namespace Shader::ResourceManager {

    bool InitResourceManager() {
        if(Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Cannot Be Initialized Twice");
            return false;
        }

        Shader::HostMemory::shaderFileCatalog = new FileCatalog();
        if(!Shader::HostMemory::shaderFileCatalog) {
            Shader::Logging::log(LOG_TYPE_ERROR, "Failed To Create FileCatalog Object");
            return false;
        }

        Shader::HostMemory::resourceManagerInitialized = true;
        Shader::Logging::log(LOG_TYPE_INFO,"Testing Resource Manager Initialization");
        return true;
    }

    bool ShutdownResourceManager() {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "Cannot Shutdown ResourceManager Sub-Module If It Is Not Currently Initialized!");
            return false;
        }

        if(Shader::HostMemory::shaderFileCatalog) {
            delete Shader::HostMemory::shaderFileCatalog;
        }

        if(Shader::HostMemory::shaderFileCatalog) {
            Shader::Logging::log(LOG_TYPE_ERROR, "Failed To Destroy File Catalog Object During Resource Manager Shutdown");
            return false;
        }

        return true;
    }

    void updateResources() {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return;
        }
        Shader::HostMemory::shaderFileCatalog->updateCatalog();
    }

    bool addSearchDirectory(std::filesystem::path directoryPath, uint8_t maxRecursiveDepth) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return false;
        }
        return Shader::HostMemory::shaderFileCatalog->addDirectoryAsRecursiveSearchPoint(directoryPath, maxRecursiveDepth);
    }

    bool addSearchPath(std::filesystem::path path) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return false;
        }
        return Shader::HostMemory::shaderFileCatalog->addPathAsSearchPoint(path);
    }

    std::filesystem::path getResourceByName(const std::string& name, bool caseSensitive) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return std::filesystem::path();
        }
        Shader::Logging::log(LOG_TYPE_INFO, "Searching For Resource With Name: " + name);
        return Shader::HostMemory::shaderFileCatalog->getPathByName(name, caseSensitive);
    }

    std::filesystem::path getResourceByExtension(const std::string& extension, bool caseSensitive) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return std::filesystem::path();
        }
        Shader::Logging::log(LOG_TYPE_INFO, "Searching For Resource With Extension: " + extension);
        return Shader::HostMemory::shaderFileCatalog->getPathByExtension(extension, caseSensitive);
    }

    std::filesystem::path getResourceByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return std::filesystem::path();
        }
        Shader::Logging::log(LOG_TYPE_INFO, "Searching For Resource With Name And Extension: " + name + extension);
        return Shader::HostMemory::shaderFileCatalog->getPathByNameAndExtension(name, extension, caseSensitive);
    }
    
    std::vector<std::filesystem::path> getAllResourcesByName(const std::string& name, bool caseSensitive) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return std::vector<std::filesystem::path>();
        }
        Shader::Logging::log(LOG_TYPE_INFO, "Searching For All Resources With Name: " + name);
        return Shader::HostMemory::shaderFileCatalog->getAllPathsByName(name, caseSensitive);
    }
    
    std::vector<std::filesystem::path> getAllResourcesByExtension(const std::string& extension, bool caseSensitive) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return std::vector<std::filesystem::path>();
        }
        Shader::Logging::log(LOG_TYPE_INFO, "Searching For All Resources With Extension: " + extension);
        return Shader::HostMemory::shaderFileCatalog->getAllPathsByExtension(extension, caseSensitive);
    }

    std::vector<std::filesystem::path> getAllResourcesByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive) {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return std::vector<std::filesystem::path>();
        }
        Shader::Logging::log(LOG_TYPE_INFO, "Searching For All Resources With Name And Extension: " + name + extension);
        return Shader::HostMemory::shaderFileCatalog->getAllPathsByNameAndExtension(name, extension, caseSensitive);
    }

    std::vector<std::filesystem::path> getAllAvailableResources() {
        if(!Shader::HostMemory::resourceManagerInitialized) {
            Shader::Logging::log(LOG_TYPE_ERROR, "ResourceManager Sub-Module Must Be Initialized Before Calling Any Other Methods!");
            return std::vector<std::filesystem::path>();
        }
        Shader::Logging::log(LOG_TYPE_INFO, "Retrieving All Available Resources");
        return Shader::HostMemory::shaderFileCatalog->getAllAvailablePaths();
    }

}
