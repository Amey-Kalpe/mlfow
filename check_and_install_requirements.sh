#!/bin/bash

# Check if the requirements.txt file exists
if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt file not found!"
  exit 1
fi

# Loop through each package in requirements.txt
while IFS= read -r package; do
  # Extract the package name (before any version specifier)
  package_name=$(echo $package | cut -d '=' -f 1)

  # Check if the package is installed
  if python3 -c "import pkg_resources; pkg_resources.require('$package_name')" &> /dev/null; then
    echo "$package_name is already installed."
  else
    echo "$package_name is not installed. Installing..."
    pip install "$package"
  fi
done < "requirements.txt"

echo "All requirements are satisfied."
