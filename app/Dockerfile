# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . /app

# Expose the port that Streamlit uses (8501 by default)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "schemes.py", "--server.port=8501", "--server.address=0.0.0.0"]
