import os
import urllib.request

checkpoints = [
    ("https://huggingface.co/savyak2/ren-dinov2-vitl14/resolve/main/checkpoint.pth",
     "logs/ren-dinov2-vitl14/checkpoint.pth"),
    ("https://huggingface.co/savyak2/ren-dino-vitb8/resolve/main/checkpoint.pth",
     "logs/ren-dino-vitb8/checkpoint.pth"),
]

for url, save_path in checkpoints:
    if os.path.exists(save_path):
        print(f"Already exists: {save_path}")
        continue
    print(f"Downloading {save_path} ...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  -> Saved {size_mb:.1f} MB to {save_path}")

print("Done.")
