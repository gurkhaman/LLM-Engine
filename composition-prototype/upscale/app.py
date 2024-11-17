from fastapi import FastAPI, UploadFile, File
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch
from io import BytesIO

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading the upscaling model...")
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline = pipeline.to(device)
print("Model loaded successfully!")


def load_image(file: UploadFile):
    """
    Load an image from the uploaded file.
    """
    try:
        return Image.open(BytesIO(file.file.read())).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def upscale_image_with_pipeline(image: Image, prompt: str = "a generic vehicle"):
    """
    Upscale the given image using the Stable Diffusion Upscale Pipeline.
    """
    try:
        upscaled_image = pipeline(prompt=prompt, image=image).images[0]
        return upscaled_image
    except Exception as e:
        raise RuntimeError(f"Upscaling failed: {str(e)}")


def image_to_bytes(image: Image):
    """
    Convert an image to bytes for response.
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


### MAIN ENDPOINT ###


@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...)):
    """
    Endpoint to upscale a received image.
    Accepts a low-resolution image, upscales it, and returns the result.
    """
    try:
        # Load the low-resolution image
        low_res_image = load_image(file)
        print("Image loaded successfully.")

        # Perform upscaling
        prompt = "a generic vehicle"
        upscaled_image = upscale_image_with_pipeline(low_res_image, prompt)
        print("Image upscaled successfully.")

        # Convert upscaled image to bytes for returning
        img_byte_arr = image_to_bytes(upscaled_image)

        # Return the image as bytes
        return {
            "status": "success",
            "message": "Image upscaled successfully.",
            "image_bytes": img_byte_arr.getvalue(),
        }

    except ValueError as ve:
        return {"status": "error", "message": f"Image loading error: {str(ve)}"}
    except RuntimeError as re:
        return {"status": "error", "message": f"Upscaling error: {str(re)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}
