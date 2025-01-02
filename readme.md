# Ray Tracing Project

This project is a real-time ray tracing application built using the following technologies:

- **SDL**: Handles window creation and image display. SDL (and SDL_ttf) are included as part of the dependencies in the `libs/` folder.
- **CUDA**: Responsible for GPU acceleration and the core ray tracing computations.
- **stb_image_write**: Enables saving the rendered image as an HDR file.
- **CMake**: Simplifies the build process with a modular, cross-platform build system.
- **Python**: Provides a convenient build script (`Build.py`) to automate the build and run process.

## Features
- Real-time ray tracing powered by CUDA for high performance.
- A simple and effective build process using CMake or the provided Python script.
- HDR image output for high-quality renders.

## Getting Started

### Prerequisites
Ensure you have the following installed:

- **CUDA Toolkit** (for GPU acceleration)
- **CMake** (for the build system)
- **Python 3.x** (for the `Build.py` script)
- A compatible C++ compiler (e.g., GCC, Clang, or MSVC)

### Building the Project

#### Option 1: Automated Build with `Build.py`
The `Build.py` script handles everything:

1. Run `python Build.py` from the root directory.
2. The script will configure and build the project using CMake and run the resulting executable automatically.

#### Option 2: Manual Build with CMake
You can manually build the project with the following steps:

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```
2. Run CMake to generate the build system:
   ```bash
   cmake ..
   ```
3. Build the project:
   ```bash
   cmake --build .
   ```
4. Run the executable:
   ```bash
   ./Debug/RayTracing
   ```

### Output
- The rendered image is displayed in a window.
- Optionally, the image is saved as an HDR file in the project directory.

## Project Structure
```
RayTracingProject/
├── Build.py             # Python script to automate the build and run process
├── CMakeLists.txt       # CMake configuration file
├── libs/                # Contains SDL and SDL_ttf dependencies
├── src/                 # Source files for the project, including ray tracing logic and shaders
├── include/             # Header files
└── README.md            # Project documentation
```

## Usage
Feel free to use or modify this project in any way you wish. No license currently applies.

## Contributing
Contributions are welcome! If you'd like to contribute, feel free to fork the repository and submit a pull request.

