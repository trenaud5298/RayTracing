#include <Shader/Shader_Logging.h>

namespace Shader {
    namespace Settings {

        bool InitSettings() {
            Shader::Logging::log(LOG_TYPE_INFO, "Settings Sub-Module Initialized");
            return true;
        }
    }
}