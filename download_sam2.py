import os
import urllib.request

url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
save_path = "checkpoints/sam2.1_hiera_large.pt"

if os.path.exists(save_path):
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Already exists: {save_path} ({size_mb:.1f} MB)")
else:
    print(f"Downloading SAM2.1 Hiera Large...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  -> Saved {size_mb:.1f} MB to {save_path}")

print("Done.")
