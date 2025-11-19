import io
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.api import app

client = TestClient(app)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def test_image_bytes():
    """Create a simple in-memory RGB image."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def tmp_outputs_dir(tmp_path, monkeypatch):
    """Patch the outputs directory to use a temporary path."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    
    # Patch Path in the api module to redirect "outputs" to temp directory
    original_path = Path
    
    def custom_path(x):
        if x == "outputs":
            return outputs
        return original_path(x)
    
    monkeypatch.setattr("api.api.Path", custom_path)
    return outputs


# -----------------------------
# Home page
# -----------------------------
def test_home_page():
    """Test that home page returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# -----------------------------
# Predict endpoint
# -----------------------------
def test_predict_endpoint(test_image_bytes):
    """Test predict endpoint with real image."""
    files = {"file": ("test.jpg", test_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["dog", "cat", "frog", "horse"]
    assert data["filename"] == "test.jpg"


@patch("api.api.predict", return_value="dog")  # ← Patch donde se USA, no donde se define
def test_predict_endpoint_mocked(mock_predict, test_image_bytes):
    """Test predict endpoint with mocked prediction."""
    files = {"file": ("test.jpg", test_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    data = response.json()
    
    assert response.status_code == 200
    assert data["predicted_class"] == "dog"
    assert data["filename"] == "test.jpg"
    mock_predict.assert_called_once()


def test_predict_endpoint_invalid_file():
    """Test predict endpoint with invalid file."""
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "error" in data


# -----------------------------
# Resize endpoint
# -----------------------------
def test_resize_endpoint_fixed_size(test_image_bytes, tmp_outputs_dir):
    """Test resize with explicit width and height."""
    files = {"file": ("test.jpg", test_image_bytes, "image/jpeg")}
    response = client.post("/resize", files=files, data={"width": 50, "height": 60})
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    img = Image.open(io.BytesIO(response.content))
    assert img.size == (50, 60)


def test_resize_endpoint_random_size(test_image_bytes, tmp_outputs_dir):
    """Test resize with random dimensions (no width/height provided)."""
    files = {"file": ("test.jpg", test_image_bytes, "image/jpeg")}
    response = client.post("/resize", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    img = Image.open(io.BytesIO(response.content))
    # Should be random between 28-225
    assert 28 <= img.size[0] <= 225
    assert 28 <= img.size[1] <= 225


@patch("api.api.resize")  # ← Patch donde se USA
def test_resize_endpoint_mocked(mock_resize, test_image_bytes, tmp_outputs_dir):
    """Test resize endpoint with mocked resize function."""
    # Create a mock image to return
    mock_img = Image.new("RGB", (80, 80))
    mock_resize.return_value = mock_img
    
    files = {"file": ("test.jpg", test_image_bytes, "image/jpeg")}
    response = client.post("/resize", files=files)
    
    assert response.status_code == 200
    img = Image.open(io.BytesIO(response.content))
    assert img.size == (80, 80)
    mock_resize.assert_called_once()


def test_resize_endpoint_invalid_dimensions(test_image_bytes, tmp_outputs_dir):
    """Test resize with invalid dimensions."""
    files = {"file": ("test.jpg", test_image_bytes, "image/jpeg")}
    # Negative dimensions should cause an error
    response = client.post("/resize", files=files, data={"width": -50, "height": 60})
    
    # The endpoint should handle this gracefully
    assert response.status_code == 200
    data = response.json()
    assert "error" in data


# -----------------------------
# Get output file endpoint
# -----------------------------
def test_get_output_file_existing(tmp_outputs_dir):
    """Test retrieving an existing file from outputs."""
    file_path = tmp_outputs_dir / "sample.jpg"
    Image.new("RGB", (50, 50)).save(file_path)

    response = client.get(f"/outputs/{file_path.name}")
    assert response.status_code == 200
    img = Image.open(io.BytesIO(response.content))
    assert img.size == (50, 50)


def test_get_output_file_missing(tmp_outputs_dir):
    """Test retrieving a non-existent file."""
    response = client.get("/outputs/nonexistent.jpg")
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "File not found"