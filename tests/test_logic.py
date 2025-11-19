import pytest
import random
from unittest.mock import patch
from pathlib import Path
from PIL import Image

from logic.utilities import (
    predict,
    resize,
    to_grayscale,
    normalize,
    random_rotate,
    blur,
    random_flip,
    preprocess,
    ensure_output_dir,
)

# Utility to generate dummy image
@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)
    return img_path


# ─────────────────────────────
# PREDICT
# ─────────────────────────────
def test_predict_returns_valid_class():
    classes = ["cat", "dog", "frog", "horse"]
    for _ in range(10):
        result = predict(None)
        assert result in classes


def test_predict_not_empty_classes():
    with patch("logic.utilities.random.choice") as mock_choice:
        predict(None)
        mock_choice.assert_called_once()


# ─────────────────────────────
# RESIZE
# ─────────────────────────────
def test_resize_specific_dimensions(dummy_image):
    img = resize(dummy_image, width=50, height=60)
    assert img.size == (50, 60)


@patch("logic.utilities.random.randint", return_value=100)
def test_resize_random_dimensions(mock_rand, dummy_image):
    img = resize(dummy_image)
    assert img.size == (100, 100)


# ─────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────
def test_to_grayscale(dummy_image):
    img = Image.open(dummy_image)
    gray = to_grayscale(img)
    assert gray.mode == "L"  # grayscale mode


def test_normalize_grayscale(dummy_image, tmp_path):
    """
    Ensure normalize() correctly normalizes grayscale images
    so the 'else' branch (non-tuple pixel) is executed.
    """
    # Create a grayscale version of the dummy image
    gray_path = tmp_path / "gray.jpg"
    Image.open(dummy_image).convert("L").save(gray_path)

    img = Image.open(gray_path)
    normalized = normalize(img)

    # Pixel should NOT be a tuple → triggers the `else: assert 0 <= px <= 1`
    px = normalized.getpixel((0, 0))
    assert not isinstance(px, tuple)
    assert 0 <= px <= 1



@patch("logic.utilities.random.uniform", return_value=10)
def test_random_rotate(mock_rot, dummy_image):
    img = Image.open(dummy_image)
    rotated = random_rotate(img)
    mock_rot.assert_called_once()
    assert isinstance(rotated, Image.Image)


@patch("logic.utilities.ImageOps.mirror")
@patch("logic.utilities.random.random", return_value=0.9)
def test_random_flip_flips(mock_rand, mock_mirror, dummy_image):
    img = Image.open(dummy_image)
    random_flip(img)
    mock_mirror.assert_called_once()


@patch("logic.utilities.random.random", return_value=0.1)
def test_random_flip_no_flip(mock_rand, dummy_image):
    img = Image.open(dummy_image)
    result = random_flip(img)
    assert result == img


def test_blur(dummy_image):
    img = Image.open(dummy_image)
    blurred = blur(img)
    assert isinstance(blurred, Image.Image)


# ─────────────────────────────
# PREPROCESS PIPELINE
# ─────────────────────────────
@patch("logic.utilities.resize")
@patch("logic.utilities.blur")
@patch("logic.utilities.random_flip")
@patch("logic.utilities.random_rotate")
@patch("logic.utilities.to_grayscale")
def test_preprocess_call_order(
    mock_gray, mock_rot, mock_flip, mock_blur, mock_resize, dummy_image
):
    mock_resize.return_value = Image.new("RGB", (64, 64))
    mock_gray.return_value = Image.new("L", (64, 64))
    mock_rot.return_value = Image.new("L", (64, 64))
    mock_flip.return_value = Image.new("L", (64, 64))
    mock_blur.return_value = Image.new("L", (64, 64))

    output = preprocess(dummy_image)
    assert isinstance(output, Image.Image)

    mock_resize.assert_called_once()
    mock_gray.assert_called_once()
    mock_rot.assert_called_once()
    mock_flip.assert_called_once()
    mock_blur.assert_called_once()


# ─────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────
def test_ensure_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = ensure_output_dir()
    assert out.exists()
    assert out.is_dir()
