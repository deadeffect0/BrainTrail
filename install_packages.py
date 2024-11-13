import subprocess

# Define the path to your requirements.txt file
requirements_file = 'requirements.txt'

# Install the packages listed in requirements.txt
subprocess.call(['pip', 'install', '-r', requirements_file])

print("All packages from requirements.txt have been installed.")
