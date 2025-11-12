import torch
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2  # Import OpenCV
import numpy as np

# --- Step 1: Re-define the Model Architecture (Same as before) ---
class CLIPImageClassifier(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super(CLIPImageClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.clip.config.vision_config.hidden_size, 256),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.pooler_output
        return self.classifier(image_features)


# --- Step 2: Load Model and Processor (Same as before) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_clip_finetuned_classifier.pth"
clip_model_name = "openai/clip-vit-base-patch32"

model = CLIPImageClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)), strict=False)
model.eval()
processor = CLIPProcessor.from_pretrained(clip_model_name)
print("Model and processor loaded.")


# --- Step 3: Modify Inference Function to Accept PIL Images ---
# This is slightly different from the previous function that took a file path
def predict_frame(pil_image, model, processor, device):
    """Takes a PIL Image and returns the model's prediction."""
    # Process the PIL image
    inputs = processor(images=pil_image, return_tensors="pt")['pixel_values'].to(device)

    # Perform inference
    with torch.no_grad():
        output = model(inputs)
    
    # Get the probability score (0 = Real, 1 = Fake)
    probability = output.item()
    return probability


# --- Step 4: NEW - Video Inference Function ---
def predict_video(video_path, model, processor, device, frames_to_sample=1):
    """
    Analyzes a video file frame by frame and returns an aggregated prediction.
    frames_to_sample: Number of frames to check per second.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video's frames per second (fps)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps / frames_to_sample)) # Calculate how many frames to skip
    
    frame_count = 0
    predictions = []

    print(f"Analyzing video... sampling 1 frame every {frame_skip} frames.")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break  # End of video

        # Sample frames to save computation
        if frame_count % frame_skip == 0:
            # 1. Convert OpenCV frame (BGR NumPy array) to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 2. Get prediction for the frame
            fake_probability = predict_frame(pil_image, model, processor, device)
            predictions.append(fake_probability)
        
        frame_count += 1

    video.release()

    # --- Step 5: Aggregate Results ---
    if not predictions:
        print("No frames were analyzed.")
        return

    # Calculate the average "fake" probability across all sampled frames
    avg_fake_probability = sum(predictions) / len(predictions)
    
    final_verdict = "Fake" if avg_fake_probability > 0.5 else "Real"
    
    print("\n--- Video Analysis Complete ---")
    print(f"Total frames sampled: {len(predictions)}")
    print(f"Average 'Fake' Probability: {avg_fake_probability:.4f}")
    print(f"Final Video Verdict: {final_verdict}")
    return final_verdict, avg_fake_probability


# --- Step 6: Example Usage ---
test_video_path = "fake.mp4"
predict_video(test_video_path, model, processor, DEVICE, frames_to_sample=1)