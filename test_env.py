from dotenv import load_dotenv
import os

#load_dotenv()



from loadotenv import load_env # removed in GCP deployment
load_env() # This will be removed for the GCP deployment
#wandb_api_key = os.environ.get('WANDB_API_KEY')


api_key = os.getenv("WANDB_API_KEY")
print(f"API_KEY is {api_key}")

