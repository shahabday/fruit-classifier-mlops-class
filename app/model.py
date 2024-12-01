import os
import wandb
from loadotenv import load_env
from pathlib import Path


load_env()
wandb_api_key = os.environ.get('WANDB_API_KEY')
print(wandb_api_key)

MODELS_DIR = 'models'

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    #assert 'WANDB_API_KEY' in os.environ, 'Please enter the wandb API key'

    wandb_org = os.environ.get('WANDB_ORG')
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')
    wandb_full_path = os.environ.get('FULL_MODEL_PATH') # this is just for debugging

    # This one might work (?)
    #artifact_path = str(Path(wandb_org) / wandb_project / wandb_model_name) + (f':{wandb_model_version}')
    print(wandb_full_path)
    artifact_path = wandb_full_path
    
    wandb.login()
    artifact = wandb.Api().artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)

    print(artifact_path)

download_artifact()