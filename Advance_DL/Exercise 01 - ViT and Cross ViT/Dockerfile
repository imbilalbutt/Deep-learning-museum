# Use an official Python runtime as a parent image
FROM python:3.9 AS model_training

# Set the working directory in the container
WORKDIR /app

# Copy the model training script into the container
# copy src folder inside Container/app
#COPY src/ .
# Copy the source code for model training
#COPY src/model_training.py /app/
COPY requirements.txt /app/

# Install Scikit-Learn and joblib
RUN pip install -r requirements.txt

# Run the model training script
RUN python ex1_main.py

# Stage 2: Serve Predictions
FROM python:3.9 AS serving

WORKDIR /app

# Copy the trained model from the previous stage
COPY --from=model_training /app/my_model.keras /app/