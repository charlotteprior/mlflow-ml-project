# api/Dockerfile

# 1. Base Image
# Use a lightweight official Python image
FROM python:3.11-slim

# 2. Set Working Directory
# Set the current directory inside the container to /app/api
# The uvicorn command will run from this context
WORKDIR /app

# 3. Install Dependencies
# Copy requirements.txt from the parent directory (../requirements.txt)
# and install dependencies. This creates an isolated virtual environment.
COPY ../requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 4. Copy Application Code
# Copy app.py from the current directory (the 'api' folder)
COPY api/ /app/api/
COPY src/ /app/src/

# 5. Expose Port
# Inform Docker that the container listens on port 8000
EXPOSE 8000

# 6. Set Environment Variables
# These variables will be overridden by the 'docker run -e' command.
# The container uses these to locate the MLflow server (on the host machine).
# 
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5001"
ENV RUN_ID = "fef4dbf213f645d594d85200b3124ab6"

# 7. Default Run Command
# This command runs Uvicorn to start the FastAPI application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
