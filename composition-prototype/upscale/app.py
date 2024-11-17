from fastapi import FastAPI, UploadFile, File
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch
from io import BytesIO

