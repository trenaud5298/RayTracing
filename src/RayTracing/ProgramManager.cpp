#include <RayTracing/ProgramManager.h>

ProgramManager::ProgramManager(const std::string& programName) {
	//Initialize SDL Subsystems
	this->initializeSDLElements(SDL_INIT_VIDEO, true);

	//Initialize Main Display Window
	this->mainDisplay = new Display(1920,1080,programName);

	//Initialize Shader
	Shader::init(1920,1080);
	Shader::runKernel();
	float* rawTestImageData = Shader::getRawImageData();
	int testPixel = 1920 *1080-100;
	std::cout<<"Raw Pixel Data: ( "<<rawTestImageData[testPixel]<<", "<<rawTestImageData[testPixel+1]<<", "<<rawTestImageData[testPixel+2]<<" )"<<std::endl;

	
	this->mainDisplay->updateDisplay(rawTestImageData);
	Shader::saveImage("test.hdr");
}	


ProgramManager::~ProgramManager() {
	std::cout<<"Handling Program Cleanup"<<std::endl;
	if(this->mainDisplay) {
		delete this->mainDisplay;
	}
}

void ProgramManager::run() {
	this->programIsRunning = true;
	while (this->programIsRunning) {
		std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
		this->handleEvents();
		Shader::runKernel(100);
		this->mainDisplay->updateDisplay(Shader::getRawImageData());
		std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
		// Convert duration to FPS
		double fps = 1000000.0 / duration.count();  // If using seconds
		std::cout << "Duration: " << duration.count() << " milliseconds\n";
		std::cout << "FPS: " << fps << std::endl;
	}
}


void ProgramManager::handleEvents() {
	while (SDL_PollEvent(&this->programEvents)) {
		if (this->programEvents.type == SDL_EVENT_QUIT) {
			this->programIsRunning = false;
		}
	}
}




void ProgramManager::initializeSDLElements(Uint32 SDLSubsystems, bool InitSDLTTF) {
	Uint32 remainingSubsystems = SDLSubsystems;
	Uint32 currentSubsystem;
	while (remainingSubsystems != 0) {
		// Extract the lowest bit set
		currentSubsystem = remainingSubsystems & (~(remainingSubsystems - 1));

		// Check if the subsystem is already initialized
		if ((SDL_WasInit(currentSubsystem) & currentSubsystem) == 0) {
			// Attempt to initialize the subsystem
			if (SDL_InitSubSystem(currentSubsystem) != true) {
				throw std::runtime_error("Failed To Initialize SDL Subsystem: " + std::string(SDL_GetError()));
			}
			else {
				std::cout << "Initialized SDL Subsystem: " << currentSubsystem << std::endl;
			}
		}

		// Remove the current subsystem from the remaining mask
		remainingSubsystems &= ~currentSubsystem;
	}

	if(InitSDLTTF && TTF_WasInit() == 0) {
		if(TTF_Init() != true) {
			throw std::runtime_error("Failed To Initialize SDL_ttf: " + std::string(SDL_GetError()));
		} else {
			std::cout<<"Initialized SDL_ttf!"<<std::endl;
		}
	}
}