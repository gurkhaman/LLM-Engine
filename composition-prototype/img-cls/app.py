from fastapi import FastAPI, UploadFile, File
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load DETR model and processor
print("Loading DETR model...")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm"
)
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Model loaded successfully!")

OBJECT_DETECTION_SERVICE_URL = ""


def load_image(file: UploadFile):
    """
    Load an image from the uploaded file.
    """
    try:
        return Image.open(BytesIO(file.file.read())).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def classify_image(image: Image):
    """
    Classify objects in the image using DETR.
    Returns detected objects with their scores, labels, and bounding boxes.
    """
    # Prepare the input
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process detections
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    return results


def notify_object_detection_service(message: dict):
    """
    Send a message to the Object Detection Service.
    """
    try:
        response = requests.post(OBJECT_DETECTION_SERVICE_URL, json=message)
        print(f"Notification sent. Response: {response.status_code}, {response.json()}")
    except Exception as e:
        print(f"Failed to notify Object Detection Service: {str(e)}")

### MAIN ENDPOINT ###

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    """
    Endpoint to classify objects in an image and check for "ambulance".
    """
    try:
        # Load the image
        image = load_image(file)
        print("Image loaded successfully.")

        # Classify objects in the image
        results = classify_image(image)
        print("Classification completed.")

        # Check for "ambulance"
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = model.config.id2label[label.item()]
            if label_name == "ambulance":  # Modify if your model uses a different label
                print(f"Ambulance detected with confidence {score.item():.3f} at location {box.tolist()}")

                # Notify the Object Detection Service
                notify_object_detection_service({
                    "detected_object": "ambulance",
                    "confidence": round(score.item(), 3),
                    "bounding_box": [round(i, 2) for i in box.tolist()]
                })

                # Return a success response
                return {
                    "status": "success",
                    "message": "Ambulance detected and notification sent.",
                    "confidence": round(score.item(), 3),
                    "bounding_box": [round(i, 2) for i in box.tolist()]
                }

        # If no ambulance was detected
        return {"status": "success", "message": "No ambulance detected."}

    except ValueError as ve:
        return {"status": "error", "message": f"Image loading error: {str(ve)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}