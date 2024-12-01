import os
import wandb
from loadotenv import load_env
from pathlib import Path
import torch
from torchvision.models import resnet18, ResNet
from torch import nn
from torchvision import transforms


load_env()
wandb_api_key = os.environ.get('WANDB_API_KEY')
print(wandb_api_key)

MODELS_DIR = 'models'
MODEL_FILE_NAME = 'best_model.pth'

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    #assert 'WANDB_API_KEY' in os.environ, 'Please enter the wandb API key'

    wandb_org = os.environ.get('WANDB_ORG')
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

    artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"

    wandb.login()
    artifact = wandb.Api().artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)

def get_raw_model() -> ResNet:
    """Here we create a model with the same architecture as the one that we have on Kaggle, but without any weights"""
    architecture = resnet18(weights=None)
    # Change the model architecture to the one that we are actually using 
    architecture.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 6)
    )

    return architecture 

print(get_raw_model())