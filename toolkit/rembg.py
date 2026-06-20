from pathlib import Path
from rembg import remove

target_dir = r"" # Image folder to process

for png_path in Path(target_dir).glob("*.png"):
    with open(png_path, 'rb') as f:
        input_img = f.read()
    output_img = remove(input_img)
    with open(png_path, 'wb') as f:
        f.write(output_img)
    print(f"Finished: {png_path.name}")
print("Done")