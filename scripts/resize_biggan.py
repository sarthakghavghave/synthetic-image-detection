import os
import sys
from PIL import Image
from tqdm import tqdm

from pathlib import Path

PROJECT_ROOT = Path().resolve().parents[0]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Source and destination
src_root = PROJECT_ROOT / "Datasets/GenImage/BigGAN/imagenet_ai_0419_biggan/val"
dst_root = PROJECT_ROOT / "Datasets/GenImage/BigGAN/BigGAN_rs"

# Folders for real/fake
classes = ["ai", "nature"]
resize_to = (224, 224)

for cls in classes:
    src_folder = os.path.join(src_root, cls)
    dst_folder = os.path.join(dst_root, cls)
    os.makedirs(dst_folder, exist_ok=True)

    for fname in tqdm(os.listdir(src_folder), desc=f"Processing {cls}"):
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, os.path.splitext(fname)[0] + ".jpg")

        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize(resize_to, Image.LANCZOS)
            img.save(dst_path, "JPEG", quality=90)
        except:
            continue
