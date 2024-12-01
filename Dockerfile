# Sets the Python version for the container
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file from our folder into /code folder in the container
COPY ./requirements.txt /code/requirements.txt

# Install the requirements.txt file inside the container
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the code inside the app folder in our repository to the code/app folder in the container
COPY ./app /code/app

# Set environment variables
ENV WANDB_API_KEY=""
ENV WANDB_ORG=""
ENV WANDB_PROJECT=""
ENV WANDB_MODEL_NAME=""
ENV WANDB_MODEL_VERSION=""


EXPOSE 8080

# Command to run the application
# same as fastapi run app/main.py --port 8080 

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
