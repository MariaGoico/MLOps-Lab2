import click
from logic.utilities import (
    predict,
    resize,
    to_grayscale,
    normalize,
    random_rotate,
    blur,
    random_flip,
    preprocess as preprocess_pipeline,
    ensure_output_dir,
)
from PIL import Image
from pathlib import Path

# ─────────────────────────────
# MAIN GROUP
# ─────────────────────────────
@click.group(help="Main CLI tool for image classification utilities.")
def cli():
    """
    Main entry point of the CLI.

    Commands:
        - classify
        - preprocess
    """
    pass


# ─────────────────────────────
# CLASSIFICATION GROUP
# ─────────────────────────────
@cli.group(help="Commands related to classification.")
def classify():
    """Group for prediction-related commands."""
    pass


@classify.command(
    name="predict",
    help="Predict a random class for an image (using hardcoded classes: dog, cat, frog, horse). "
         "Example: python -m cli.cli classify predict image.jpg"
)
@click.argument("image_path")
def classify_predict(image_path):
    """
    Predict a random class for a given image.
    Classes are hardcoded for testing: dog, cat, frog, horse.

    Example:
        python -m cli.cli classify predict image.jpg
    """
    # Validate image exists
    img_file = Path(image_path)
    if not img_file.exists():
        click.echo(f"Error: Image file '{image_path}' not found.", err=True)
        return
    
    # Hardcoded classes (same as API)
    classes = ['dog', 'cat', 'frog', 'horse']
    
    # Predict
    result = predict(classes)
    click.echo(f"Image: {image_path}")
    click.echo(f"Predicted class: {result}")


# ─────────────────────────────
# PREPROCESSING GROUP
# ─────────────────────────────
@cli.group(help="Commands related to preprocessing images.")
def preprocess():
    """Group for preprocessing commands."""
    pass


@preprocess.command(
    name="resize",
    help="Resize an image to a specified size (or random 28-225 if not specified). "
         "Example: python -m cli.cli preprocess resize input.jpg"
)
@click.argument("image_path")
@click.option("--width", type=int, default=None, help="Target width (random if not specified)")
@click.option("--height", type=int, default=None, help="Target height (random if not specified)")
@click.option("--output", "-o", default=None, help="Output path (default: outputs/resized_<filename>)")
def preprocess_resize(image_path, width, height, output):
    """
    Resize an image and save the result.

    Args:
        image_path (str): Input image.
        width (int): Target width.
        height (int): Target height.
        output (str): Path to save resized image.
    """
    ensure_output_dir()
    
    # Generate output path if not provided
    if output is None:
        input_file = Path(image_path)
        output = f"outputs/resized_{input_file.name}"
    
    img = resize(image_path, width, height)
    img.save(output)
    click.echo(f"Saved resized image to: {output}")


@preprocess.command(
    name="grayscale",
    help="Convert image to grayscale."
)
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path (default: outputs/grayscale_<filename>)")
def preprocess_grayscale(image_path, output):
    """Convert image to grayscale."""
    ensure_output_dir()
    
    if output is None:
        input_file = Path(image_path)
        output = f"outputs/grayscale_{input_file.name}"
    
    img = Image.open(image_path)
    img = to_grayscale(img)
    img.save(output)
    click.echo(f"Saved grayscale image to: {output}")


@preprocess.command(
    name="rotate",
    help="Randomly rotate the image by up to ±20 degrees."
)
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path (default: outputs/rotated_<filename>)")
def preprocess_rotate(image_path, output):
    """Randomly rotate image."""
    ensure_output_dir()
    
    if output is None:
        input_file = Path(image_path)
        output = f"outputs/rotated_{input_file.name}"
    
    img = Image.open(image_path)
    img = random_rotate(img)
    img.save(output)
    click.echo(f"Saved rotated image to: {output}")


@preprocess.command(
    name="flip",
    help="Randomly flip the image horizontally (50% probability)."
)
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path (default: outputs/flipped_<filename>)")
def preprocess_flip(image_path, output):
    """Randomly flip image horizontally."""
    ensure_output_dir()
    
    if output is None:
        input_file = Path(image_path)
        output = f"outputs/flipped_{input_file.name}"
    
    img = Image.open(image_path)
    img = random_flip(img)
    img.save(output)
    click.echo(f"Saved flipped image to: {output}")


@preprocess.command(
    name="blur",
    help="Apply Gaussian blur to the image."
)
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path (default: outputs/blurred_<filename>)")
def preprocess_blur(image_path, output):
    """Apply Gaussian blur."""
    ensure_output_dir()
    
    if output is None:
        input_file = Path(image_path)
        output = f"outputs/blurred_{input_file.name}"
    
    img = Image.open(image_path)
    img = blur(img)
    img.save(output)
    click.echo(f"Saved blurred image to: {output}")


@preprocess.command(
    name="pipeline",
    help="Apply full preprocessing pipeline (resize, grayscale, rotate, flip, blur)."
)
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path (default: outputs/processed_<filename>)")
def preprocess_full_pipeline(image_path, output):
    """Apply full preprocessing pipeline."""
    ensure_output_dir()
    
    if output is None:
        input_file = Path(image_path)
        output = f"outputs/processed_{input_file.name}"
    
    img = preprocess_pipeline(image_path)
    img.save(output)
    click.echo(f"Saved fully preprocessed image to: {output}")


# ─────────────────────────────
# ENTRY POINT
# ─────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    cli()
