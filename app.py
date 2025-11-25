import gradio as gr
import requests
from PIL import Image
import io

# Base URL of your FastAPI deployment
API_URL = "https://goico-mlops-lab2-latest.onrender.com"

# Function to call /predict
def predict_image(image):
    try:
        # Convert PIL image to bytes
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        files = {"file": ("image.jpg", buf, "image/jpeg")}
        
        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "predicted_class" in data:
            return data["predicted_class"]
        else:
            return data.get("error", "Unknown error")
    except Exception as e:
        return str(e)

# Function to call /resize
def resize_image(image, width, height):
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        files = {"file": ("image.jpg", buf, "image/jpeg")}
        
        # Form data for width/height
        data = {}
        if width is not None:
            data["width"] = int(width)
        if height is not None:
            data["height"] = int(height)
        
        response = requests.post(f"{API_URL}/resize", files=files, data=data, timeout=10)
        
        # If the response is JSON (error), show error
        content_type = response.headers.get("content-type")
        if "application/json" in content_type:
            return None, response.json().get("error", "Unknown error")
        
        # Else, return the resized image
        img = Image.open(io.BytesIO(response.content))
        return img, None
    except Exception as e:
        return None, str(e)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Prediction and Resizing API Demo")

    with gr.Tab("Predict Class"):
        img_input = gr.Image(type="pil", label="Upload Image")
        predict_btn = gr.Button("Predict")
        predict_output = gr.Textbox(label="Predicted Class")
        predict_btn.click(predict_image, inputs=img_input, outputs=predict_output)

    with gr.Tab("Resize Image"):
        img_input_resize = gr.Image(type="pil", label="Upload Image")
        width_input = gr.Number(label="Width (px, optional)", value=None)
        height_input = gr.Number(label="Height (px, optional)", value=None)
        resize_btn = gr.Button("Resize")
        resize_output = gr.Image(label="Resized Image")
        resize_error = gr.Textbox(label="Error", interactive=False)
        resize_btn.click(resize_image, 
                         inputs=[img_input_resize, width_input, height_input], 
                         outputs=[resize_output, resize_error])

demo.launch()
