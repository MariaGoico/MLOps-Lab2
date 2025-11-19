import pytest
from click.testing import CliRunner
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
import io

from cli.cli import cli


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def runner():
    """Create a CliRunner instance for testing Click commands."""
    return CliRunner()


@pytest.fixture
def test_image(tmp_path):
    """Create a temporary test image."""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_outputs_dir(tmp_path, monkeypatch):
    """Mock the outputs directory to use a temporary path."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    
    # Mock ensure_output_dir to create temp outputs directory
    def mock_ensure():
        outputs.mkdir(exist_ok=True)
        return outputs
    
    monkeypatch.setattr("cli.cli.ensure_output_dir", mock_ensure)
    
    # Mock the default output path construction
    original_path = Path
    
    def custom_path(x):
        if isinstance(x, str) and x.startswith("outputs/"):
            return outputs / x.replace("outputs/", "") # pragma: no cover 
        return original_path(x)
    
    monkeypatch.setattr("cli.cli.Path", custom_path)
    
    return outputs


# ═════════════════════════════════════════════════════════════
# CLASSIFY GROUP TESTS
# ═════════════════════════════════════════════════════════════

# -----------------------------
# classify predict command
# -----------------------------
def test_classify_predict_success(runner, test_image):
    """Test successful prediction with valid image."""
    result = runner.invoke(cli, ["classify", "predict", str(test_image)])
    
    assert result.exit_code == 0
    assert "Predicted class:" in result.output
    assert str(test_image) in result.output
    # Check that it's one of the hardcoded classes
    assert any(cls in result.output for cls in ["dog", "cat", "frog", "horse"])


def test_classify_predict_nonexistent_image(runner):
    """Test prediction with non-existent image file."""
    result = runner.invoke(cli, ["classify", "predict", "nonexistent.jpg"])
    
    assert result.exit_code == 0
    assert "Error: Image file 'nonexistent.jpg' not found" in result.output


@patch("cli.cli.predict", return_value="dog")
def test_classify_predict_mocked(mock_predict, runner, test_image):
    """Test prediction with mocked predict function."""
    result = runner.invoke(cli, ["classify", "predict", str(test_image)])
    
    assert result.exit_code == 0
    assert "Predicted class: dog" in result.output
    mock_predict.assert_called_once()


# ═════════════════════════════════════════════════════════════
# PREPROCESS GROUP TESTS
# ═════════════════════════════════════════════════════════════

# -----------------------------
# preprocess resize command
# -----------------------------
def test_preprocess_resize_with_dimensions(runner, test_image, tmp_path):
    """Test resize with explicit width and height."""
    output_path = tmp_path / "resized.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "resize", str(test_image),
        "--width", "50",
        "--height", "60",
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify dimensions
    img = Image.open(output_path)
    assert img.size == (50, 60)


def test_preprocess_resize_random_size(runner, test_image, tmp_path):
    """Test resize with random dimensions (no width/height specified)."""
    output_path = tmp_path / "resized_random.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "resize", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify dimensions are within random range (28-225)
    img = Image.open(output_path)
    assert 28 <= img.size[0] <= 225
    assert 28 <= img.size[1] <= 225


def test_preprocess_resize_default_output(runner, test_image, mock_outputs_dir):
    """Test resize with default output path."""
    result = runner.invoke(cli, ["preprocess", "resize", str(test_image)])
    
    assert result.exit_code == 0
    assert "Saved resized image to:" in result.output


@patch("cli.cli.resize")
def test_preprocess_resize_mocked(mock_resize, runner, test_image, tmp_path):
    """Test resize with mocked resize function."""
    mock_img = Image.new("RGB", (80, 80))
    mock_resize.return_value = mock_img
    
    output_path = tmp_path / "mocked_resize.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "resize", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    mock_resize.assert_called_once()


# -----------------------------
# preprocess grayscale command
# -----------------------------
def test_preprocess_grayscale(runner, test_image, tmp_path):
    """Test grayscale conversion."""
    output_path = tmp_path / "grayscale.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "grayscale", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify it's grayscale (mode should be 'L')
    img = Image.open(output_path)
    assert img.mode == "L"


def test_preprocess_grayscale_default_output(runner, test_image, mock_outputs_dir):
    """Test grayscale with default output path."""
    result = runner.invoke(cli, ["preprocess", "grayscale", str(test_image)])
    
    assert result.exit_code == 0
    assert "Saved grayscale image to:" in result.output


# -----------------------------
# preprocess rotate command
# -----------------------------
def test_preprocess_rotate(runner, test_image, tmp_path):
    """Test random rotation."""
    output_path = tmp_path / "rotated.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "rotate", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify image was created (rotation is random, can't verify angle)
    img = Image.open(output_path)
    assert img is not None


@patch("cli.cli.random_rotate")
def test_preprocess_rotate_mocked(mock_rotate, runner, test_image, tmp_path):
    """Test rotate with mocked rotation function."""
    mock_img = Image.new("RGB", (100, 100))
    mock_rotate.return_value = mock_img
    
    output_path = tmp_path / "rotated.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "rotate", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    mock_rotate.assert_called_once()

def test_preprocess_rotate_default_output(runner, test_image, mock_outputs_dir):
    result = runner.invoke(cli, ["preprocess", "rotate", str(test_image)])
    
    assert result.exit_code == 0
    assert "Saved rotated image to:" in result.output

# -----------------------------
# preprocess flip command
# -----------------------------
def test_preprocess_flip(runner, test_image, tmp_path):
    """Test random horizontal flip."""
    output_path = tmp_path / "flipped.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "flip", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()


@patch("cli.cli.random_flip")
def test_preprocess_flip_mocked(mock_flip, runner, test_image, tmp_path):
    """Test flip with mocked flip function."""
    mock_img = Image.new("RGB", (100, 100))
    mock_flip.return_value = mock_img
    
    output_path = tmp_path / "flipped.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "flip", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    mock_flip.assert_called_once()

def test_preprocess_flip_default_output(runner, test_image, mock_outputs_dir):
    result = runner.invoke(cli, ["preprocess", "flip", str(test_image)])

    assert result.exit_code == 0
    assert "Saved flipped image to:" in result.output

# -----------------------------
# preprocess blur command
# -----------------------------
def test_preprocess_blur(runner, test_image, tmp_path):
    """Test Gaussian blur."""
    output_path = tmp_path / "blurred.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "blur", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()


@patch("cli.cli.blur")
def test_preprocess_blur_mocked(mock_blur, runner, test_image, tmp_path):
    """Test blur with mocked blur function."""
    mock_img = Image.new("RGB", (100, 100))
    mock_blur.return_value = mock_img
    
    output_path = tmp_path / "blurred.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "blur", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    mock_blur.assert_called_once()

def test_preprocess_blur_default_output(runner, test_image, mock_outputs_dir):
    result = runner.invoke(cli, ["preprocess", "blur", str(test_image)])

    assert result.exit_code == 0
    assert "Saved blurred image to:" in result.output

# -----------------------------
# preprocess pipeline command
# -----------------------------
def test_preprocess_pipeline(runner, test_image, tmp_path):
    """Test full preprocessing pipeline."""
    output_path = tmp_path / "processed.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "pipeline", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify it's grayscale (pipeline includes grayscale conversion)
    img = Image.open(output_path)
    assert img.mode == "L"


@patch("cli.cli.preprocess_pipeline")
def test_preprocess_pipeline_mocked(mock_pipeline, runner, test_image, tmp_path):
    """Test pipeline with mocked preprocessing function."""
    mock_img = Image.new("RGB", (100, 100))
    mock_pipeline.return_value = mock_img
    
    output_path = tmp_path / "processed.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "pipeline", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    mock_pipeline.assert_called_once()

def test_preprocess_pipeline_default_output(runner, test_image, mock_outputs_dir):
    result = runner.invoke(cli, ["preprocess", "pipeline", str(test_image)])

    assert result.exit_code == 0
    assert "Saved fully preprocessed image to:" in result.output

# ═════════════════════════════════════════════════════════════
# CLI GROUP STRUCTURE TESTS
# ═════════════════════════════════════════════════════════════

def test_cli_help(runner):
    """Test main CLI help message."""
    result = runner.invoke(cli, ["--help"])
    
    assert result.exit_code == 0
    assert "Main CLI tool for image classification utilities" in result.output
    assert "classify" in result.output
    assert "preprocess" in result.output


def test_classify_group_help(runner):
    """Test classify group help message."""
    result = runner.invoke(cli, ["classify", "--help"])
    
    assert result.exit_code == 0
    assert "Commands related to classification" in result.output
    assert "predict" in result.output


def test_preprocess_group_help(runner):
    """Test preprocess group help message."""
    result = runner.invoke(cli, ["preprocess", "--help"])
    
    assert result.exit_code == 0
    assert "Commands related to preprocessing" in result.output
    assert "resize" in result.output
    assert "grayscale" in result.output
    assert "rotate" in result.output
    assert "flip" in result.output
    assert "blur" in result.output
    assert "pipeline" in result.output


# ═════════════════════════════════════════════════════════════
# EDGE CASES
# ═════════════════════════════════════════════════════════════

def test_preprocess_resize_only_width(runner, test_image, tmp_path):
    """Test resize with only width specified."""
    output_path = tmp_path / "resized_width_only.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "resize", str(test_image),
        "--width", "50",
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    img = Image.open(output_path)
    assert img.size[0] == 50  # Width should be 50
    assert 28 <= img.size[1] <= 225  # Height should be random


def test_preprocess_resize_only_height(runner, test_image, tmp_path):
    """Test resize with only height specified."""
    output_path = tmp_path / "resized_height_only.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "resize", str(test_image),
        "--height", "60",
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    img = Image.open(output_path)
    assert 28 <= img.size[0] <= 225  # Width should be random
    assert img.size[1] == 60  # Height should be 60