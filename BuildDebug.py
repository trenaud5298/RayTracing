import subprocess
import os

scriptDirectory = os.path.dirname(os.path.abspath(__file__))
buildDirectory = os.path.join(scriptDirectory, "build")
debugDirectory = os.path.join(buildDirectory, "Application/Debug")
executablePath = os.path.join(debugDirectory, "Application")

print(f"The script is located in: {scriptDirectory}")
print(f"The build is located in: {buildDirectory}")
print(f"The debug is located in: {debugDirectory}")
print(f"The exe is located in: {executablePath}")

command1 = f'cmake -S "{scriptDirectory}"" -B "{buildDirectory}"'

commands = [
    f'cmake -S "{scriptDirectory}"" -B "{buildDirectory}"',
    f'cmake --build "{buildDirectory}" --config Debug ',
    f'"{executablePath}"'
]

def run_commands(commands):
    for command in commands:
        try:
            # Run the command
            result = subprocess.run(command, shell=True, check=True)
            # Print a success message (optional)
            print(f"Command succeeded: {command}")
        except subprocess.CalledProcessError as e:
            # Handle errors if a command fails
            print(f"Command failed: {command}")
            print(f"Error: {e}")

if __name__ == "__main__":
    run_commands(commands)