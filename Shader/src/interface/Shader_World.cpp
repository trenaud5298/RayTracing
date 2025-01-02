#include <Shader/Shader_World.h>
#include <Shader/Shader_Logging.h>

namespace Shader {
    namespace World {
        bool InitWorld() {
            Shader::Logging::log(LOG_TYPE_INFO, "World Sub-Module Initialized");
            return true;
        }
    }
}