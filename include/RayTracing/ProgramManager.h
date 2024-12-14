#ifndef RAYTRACING_PROGRAMMANAGER_H
#define RAYTRACING_PROGRAMMANAGER_H

#include <RayTracing/Display.h>
#include <RayTracing/Shader.cuh>
#include <chrono>

class ProgramManager {

//Methods
public:
	//Constructor That Handles Initialization Of Program
	ProgramManager(const std::string& programName);

	~ProgramManager();

	//Main Method That Is Called To Run Program
	void run();

private:
	//Method Used To Initialize SDL and SDL_ttf
	void initializeSDLElements(Uint32 SDLSubsystems, bool InitSDLTTF);

	//Method Used To Handle Event On Every Frame
	void handleEvents();

//Data
public:


private:
	bool programIsRunning;
	SDL_Event programEvents;
	Display* mainDisplay;

};



#endif