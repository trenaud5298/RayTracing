#include <iostream>
#include <RayTracing/ProgramManager.h>

int main(int argc, char* argv[]) {
	try {
		ProgramManager testProgram("test");
		testProgram.run();
	} catch (std::exception& e) {
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,"Program Error",e.what(),NULL);
	} catch(...) {
		std::cerr<<"Caught An Unknown Exception!"<<std::endl;
	}
	return 0;
}