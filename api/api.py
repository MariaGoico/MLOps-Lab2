import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi import Request
from pathlib import Path
import io
from PIL import Image

from logic.utilities import predict, resize, ensure_output_dir

app = FastAPI(
    title="API of the Lab 1 using FastAPI",
    description="API to perform preprocessing on images",
    version="1.0.0",
)

# Serve templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the HTML homepage.
    """
    return templates.TemplateResponse(request, "index.html", {"request": request})


# ---------------------------------------------------------
# Predict Endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict_class(
    file: UploadFile = File(...),
):
    """
    Predict a random class for an uploaded image.
    The classes are hardcoded for now (dog, cat, frog, horse).
    
    Args:
        file: The image file to classify
        
    Returns:
        dict: The predicted class
    """
    try:
        # Read the uploaded file to validate it's an image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Predict (randomly selects one class)
        result = predict(image)

        return {
            "predicted_class": result,
            "filename": file.filename
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
    
# ---------------------------------------------------------
# Resize Endpoint
# ---------------------------------------------------------
@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Form(None),
    height: int = Form(None),
):
    """
    Resize an uploaded image to specified dimensions.
    If width or height are not provided, random size between 28-225 is used.

    Args:
        file: The image file to resize
        width: Target width (optional, random if not provided)
        height: Target height (optional, random if not provided)
        
    Returns:
        StreamingResponse: The resized image as JPEG
    """
    try:
        # Ensure outputs directory exists
        ensure_output_dir()
        
        # Read the binary file from UploadFile
        contents = await file.read()
        
        # Open with PIL and convert to RGB
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save the uploaded image temporarily
        input_path = Path("temp_input.jpg")
        image.save(input_path, format="JPEG")

        # Apply resize using your utilities
        resized_img = resize(str(input_path), width, height)

        # Save to outputs folder
        output_path = Path("outputs") / f"resized_{file.filename}"
        resized_img.save(output_path, format="JPEG")

        # Create a buffer for streaming response
        img_bytes = io.BytesIO()
        resized_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Clean up temporary file
        input_path.unlink(missing_ok=True)

        # Return the image
        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            #headers={"Content-Disposition": f"attachment; filename=resized_{file.filename}"}
        )

    except Exception as e:
        return {"error": f"Could not process image: {str(e)}"}


# ---------------------------------------------------------
# Get Output File Endpoint (Optional - to retrieve saved files)
# ---------------------------------------------------------
@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """
    Retrieve a file from the outputs directory.
    
    Args:
        filename: Name of the file to retrieve
        
    Returns:
        FileResponse: The requested file
    """
    file_path = Path("outputs") / filename
    
    if not file_path.exists():
        return {"error": "File not found"}
    
    return FileResponse(file_path)


# Entry point (for direct execution only)
if __name__ == "__main__":  # pragma: no cover
    uvicorn.run("api.api:app", host="localhost", port=8000, reload=True)