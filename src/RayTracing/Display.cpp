#include <RayTracing/Display.h>


Display::Display() : Display(Display::DEFAULT_RESOLUTION_X,Display::DEFAULT_RESOLUTION_Y,"Blank Display") {

}


Display::Display(const std::string& displayName) : Display(Display::DEFAULT_RESOLUTION_X, Display::DEFAULT_RESOLUTION_Y, displayName) {

}

Display::Display(const int& resolutionX, const int& resolutionY, const std::string& displayName) : 
resolutionX(resolutionX), resolutionY(resolutionY), displayName(displayName) {

	//Creates The SDL Window For Display Instance
	this->displayWindow = SDL_CreateWindow(
		this->displayName.c_str(),
		this->resolutionX,
		this->resolutionY,
		SDL_WINDOW_RESIZABLE
	);
	//Handles Window Creation Failure
	if (this->displayWindow == nullptr) {
		std::string SDL_Error_Message = SDL_GetError();
		throw std::runtime_error("Display Window Could Not Be Created! SDL_Error: " + SDL_Error_Message);
	}


	//Creates The SDL Renderer For Display Instance
	this->displayRenderer = SDL_CreateRenderer(this->displayWindow, NULL);
	//Handles Renderer Creation Failure
	if (this->displayRenderer == nullptr) {
		std::string SDL_Error_Message = SDL_GetError();
		throw std::runtime_error("Display Renderer Could Not Be Created! SDL_Error: " + SDL_Error_Message);
	}

	this->displayTexture = SDL_CreateTexture(
		this->displayRenderer,
		SDL_PIXELFORMAT_RGB96_FLOAT,
		SDL_TEXTUREACCESS_STREAMING,
		this->resolutionX,this->resolutionY
	);
	if(this->displayTexture == nullptr) {
		std::string SDL_Error_Message = SDL_GetError();
		throw std::runtime_error("Display Texture Could Not Be Created! SDL_Error: " + SDL_Error_Message);
	}
}

Display::~Display() {
	std::cout << "Destroying Display" << std::endl;
	if (this->displayWindow) {
		SDL_DestroyWindow(this->displayWindow);
	}

	if (this->displayRenderer) {
		SDL_DestroyRenderer(this->displayRenderer);
	}
}

void Display::updateDisplay(float* rawPixelData) {
	if (SDL_UpdateTexture(this->displayTexture, nullptr, rawPixelData, this->resolutionX * 3 * sizeof(float)) != true) {
        throw std::runtime_error("Failed To Update Texture: " + std::string(SDL_GetError()));
    }

	if(SDL_RenderTexture(this->displayRenderer,this->displayTexture,NULL,NULL) != true) {
        throw std::runtime_error("Failed To Render Texture: " + std::string(SDL_GetError()));
	}

	SDL_RenderPresent(this->displayRenderer);
}
