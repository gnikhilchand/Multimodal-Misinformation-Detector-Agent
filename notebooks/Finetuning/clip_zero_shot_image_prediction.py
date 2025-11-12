import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- Zero-Shot Prediction using a Base CLIP Model ---

# 1. Load the BASE model, not your finetuned one
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Define your image path and text prompts
image_path = "download4.jpg" 
image = Image.open(image_path)
text_prompts = ["a photograph taken with a camera", "a computer-generated image"]

# 3. Process the image and text
inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(DEVICE)

# 4. Get the model's predictions
with torch.no_grad():
    outputs = model(**inputs)

# The logits_per_image tensor gives the similarity scores
logits_per_image = outputs.logits_per_image 
probs = logits_per_image.softmax(dim=1)  # We apply softmax to get probabilities

# 5. Print the results
print(f"Zero-shot prediction for '{image_path}':")
for i, prompt in enumerate(text_prompts):
    print(f"  - Prompt: '{prompt}', Probability: {probs[0][i].item():.4f}")