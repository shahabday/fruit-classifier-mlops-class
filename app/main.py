
import torch
import io

from pydantic import BaseModel # this has nothing to do with ML models 
# BaseModel is just the parent class for everything that is strictly typed in Pydantic
from fastapi import FastAPI, Depends, UploadFile, File
from torchvision.models import ResNet
from PIL import Image
from app.model import load_model, load_transforms, CATEGORIES
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F

# This is a data model for the result 
class Result(BaseModel):
    category: str
    confidence: float


# Create the FastAPI instance
app = FastAPI()


# Debug message to check if the endpoint is working
@app.get('/')
def read_root():
    return {'message':
            'This is not supposed to be used with GET, send an image with POST'}

@app.post('/predict', response_model=Result)
async def predict(
        input_image: UploadFile = File(...),
        model: ResNet = Depends(load_model),
        transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    image = Image.open(io.BytesIO(await input_image.read()))

    # Here we delete the alpha channel, the model doesn't use it
    # and will complain if the input has it     
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Here we add a batch dimension of 1 
    image = transforms(image).unsqueeze(0)

    # This is inference mode, we don't need gradients 
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)


    category = CATEGORIES[predicted_class.item()]

    return Result(category=category, confidence=confidence.item())




