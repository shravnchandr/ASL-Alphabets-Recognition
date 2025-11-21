# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary project files
COPY ["Live Recognition.py", "."]
COPY ["MediaPipe Models/", "MediaPipe Models/"]
COPY ["Misc Models/", "Misc Models/"]

# Make port 80 available to the world outside this container (not strictly needed for this GUI app but good practice)
# EXPOSE 80

# Run Live Recognition.py when the container launches
CMD ["python", "Live Recognition.py"]
