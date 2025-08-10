#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib
import platform
from typing import Optional

from PIL import Image, ImageOps
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available, falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(device: torch.device):
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    )
    # Best-effort precision hint (safe on CPU/CUDA; harmless if not supported)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    model.to(device)
    model.eval()
    return model


def _resize_with_pad_rgba(image_rgba: Image.Image, output_size: int) -> Image.Image:
    """Resize RGBA image to a square canvas of output_size while preserving aspect ratio.
    Pads with full transparency.
    """
    if output_size <= 0:
        return image_rgba
    src_w, src_h = image_rgba.size
    if src_w == 0 or src_h == 0:
        return image_rgba
    scale = min(output_size / src_w, output_size / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image_rgba.resize((new_w, new_h), resample=Image.LANCZOS)
    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))
    offset_x = (output_size - new_w) // 2
    offset_y = (output_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def process_one(
    model,
    device: torch.device,
    image_path: str,
    out_path: str,
    image_size: int,
    output_size: Optional[int] = None,
) -> str:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    orig_size = image.size

    transform = build_transform(image_size)
    # Use the model's actual device to avoid mismatch if we fallback mid-run
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device

    input_tensor = transform(image).unsqueeze(0).to(model_device)

    with torch.inference_mode():
        try:
            output = model(input_tensor)[-1]
        except NotImplementedError as e:
            if str(model_device) == "mps":
                # Enable per-op CPU fallback for unsupported MPS ops
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                print("MPS op not implemented, falling back to CPU for inference...", file=sys.stderr)
                model.to("cpu")
                input_tensor = input_tensor.to("cpu")
                output = model(input_tensor)[-1]
            else:
                raise
        pred = output.sigmoid().detach().to("cpu")

    mask = pred[0].squeeze().clamp(0, 1)

    mask_pil = transforms.ToPILImage()(mask)
    mask_resized = mask_pil.resize(orig_size, resample=Image.BILINEAR)

    image_rgba = image.copy()
    image_rgba.putalpha(mask_resized)

    # Optional final resizing with padding (square) while keeping aspect ratio
    if output_size is not None:
        image_rgba = _resize_with_pad_rgba(image_rgba, int(output_size))

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not out_path.lower().endswith(".png"):
        out_path = out_path + ".png"
    image_rgba.save(out_path)
    return out_path


def iter_images(input_path: str):
    p = pathlib.Path(input_path)
    if p.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff", "*.jfif", "*.avif"):
            yield from p.rglob(ext)
    else:
        yield p


def main():
    parser = argparse.ArgumentParser(description="Remove background using briaai/RMBG-2.0")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file or directory. Default: alongside input(s) with suffix _rmbg.png",
    )
    parser.add_argument("--size", type=int, default=1024, help="Model input size (default 1024)")
    parser.add_argument(
        "--output-size",
        type=int,
        default=None,
        help="Optional final output size (square). If set, export will be resized with transparent padding to NxN.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run on (default: auto)",
    )
    args = parser.parse_args()

    # Proactively enable MPS fallback on macOS to survive unsupported ops
    if platform.system() == "Darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = get_device(args.device)
    model = load_model(device)

    inputs = list(iter_images(args.input))
    if not inputs:
        print("No images found", file=sys.stderr)
        sys.exit(1)

    for in_path in inputs:
        in_path = pathlib.Path(in_path)
        if args.output:
            out_base = pathlib.Path(args.output)
            if len(inputs) == 1 and (not out_base.exists() or out_base.suffix):
                out_path = out_base
            else:
                if out_base.suffix:
                    out_base = out_base.parent
                out_path = out_base / (in_path.stem + "_rmbg.png")
        else:
            out_path = in_path.with_name(in_path.stem + "_rmbg.png")

        saved = process_one(
            model,
            device,
            str(in_path),
            str(out_path),
            args.size,
            output_size=args.output_size,
        )
        print(f"Saved: {saved}")


if __name__ == "__main__":
    main()


