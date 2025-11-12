import torch
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- Step 1: Re-define the Model Architecture ---
# You MUST use the exact same model class definition as during training.
class CLIPImageClassifier(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super(CLIPImageClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # We don't need to freeze the weights here since we are not training
        
        self.classifier = nn.Sequential(
            nn.Linear(self.clip.config.vision_config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.pooler_output
        return self.classifier(image_features)


# --- Step 2: Load the Finetuned Weights ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_clip_finetuned_classifier.pth" # <-- Make sure this path is correct

# Instantiate the model
model = CLIPImageClassifier().to(DEVICE)

# 1. Load the state dictionary from the file
# We load it onto the CPU first to avoid GPU memory issues if the file is large
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# 2. Create a new state dictionary without the 'module.' prefix
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('module.'):
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

# Load the saved state dictionary
# model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.load_state_dict(new_state_dict, strict=False)


# Set the model to evaluation mode
# This is crucial as it disables layers like Dropout
model.eval()
print("Model loaded and set to evaluation mode.")


# --- Step 3: Create an Inference Function ---
def predict_image(image_path, model, processor, device):
    """
    Takes an image path and returns the model's prediction.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error opening image: {e}", None

    # Process the image
    inputs = processor(images=image, return_tensors="pt")['pixel_values'].to(device)

    # Perform inference
    with torch.no_grad():
        output = model(inputs)
    
    # Get the probability score
    probability = output.item()
    
    # Get the binary classification (Fake/Real) [cite: 46]
    prediction = "Fake" if probability > 0.5 else "Real"
    
    return prediction, probability


# --- Step 4: Example Usage ---
# You need the same CLIP processor used during training
clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name)

# Provide the path to a new image you want to test
# Replace this with your own image path
test_image_path = "download2.jpg" 

prediction, score = predict_image(test_image_path, model, processor, DEVICE)

if prediction:
    print(f"The image '{test_image_path}' is classified as: {prediction}")
    print(f"Confidence Score : {score:.4f}")