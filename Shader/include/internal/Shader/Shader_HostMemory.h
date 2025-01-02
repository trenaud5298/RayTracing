#ifndef SHADER_HOSTMEMORY_H
#define SHADER_HOSTMEMORY_H

#include <Shader/Shader_Log.h>
#include <Shader/Shader_FileCatalog.h>

namespace Shader::HostMemory {
#pragma region Logging

    extern bool loggingInitialized;
    extern ShaderLog* shaderLog;

#pragma endregion



#pragma region ResourceManager

    extern bool resourceManagerInitialized;
    extern FileCatalog* shaderFileCatalog;
    
#pragma endregion



#pragma region Settings

#pragma endregion



#pragma region World

#pragma endregion



#pragma region Debug

#pragma endregion



};


#endif