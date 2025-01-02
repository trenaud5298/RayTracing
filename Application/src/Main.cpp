#include <Shader/Shader.h>
#include <Shader/Shader_Logging.h>
#include <Shader/Shader_ResourceManager.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <set>
#include <Shader/Shader_Timer.h>


int main(int argc, char* argv[]) {
    
    Timer<std::chrono::milliseconds> test("Application Timer");
    Shader::Init();
    std::cout<<"Add .dll Files Found"<<"\n";
    for(const std::filesystem::path& resource : Shader::ResourceManager::getAllResourcesByExtension(".dll", false)) {
        std::cout<<resource.string()<<"\n";
    }

    std::cout<<"\n\n"<<Shader::Logging::GetLog()<<"\n";
    std::cout<<"\n\n\nProgram Complete\n\n"<<std::endl;
	return 0;
}

