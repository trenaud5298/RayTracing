#ifndef RAYTRACING_DISPLAY_H
#define RAYTRACING_DISPLAY_H

#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>

//Class That Creates A Display Instance THIS CLASS NEEDS TO BE RWORKED TO BETTER HANDLE MULTIPLE SITUATIONS
class Display {

//Methods
public:
	//Calls Final Constructor With Default Resolution And Name
	Display();

	//Calls Final Constructor With Default Resolution And Passed In Name
	Display(const std::string& displayName);

	/**
	 * Creates A Display Instance With Passed In Resolution
	 * @param resolutionX - Width Of Display
	 * @param resolutionY - Height Of Display
	 */
	Display(const int& resolutionX, const int& resolutionY, const std::string& displayName);

	//Destructor That Handles The Destruction Of Display Object
	~Display();

	void updateDisplay(float* rawPixelData);

	
private:



	//Data
	public:
		const static int DEFAULT_RESOLUTION_X = 1200;
		const static int DEFAULT_RESOLUTION_Y = 800;


	private:
		int resolutionX;
		int resolutionY;
		std::string displayName;
		SDL_Window* displayWindow;
		SDL_Renderer* displayRenderer;
		SDL_Texture* displayTexture;
	};

#endif