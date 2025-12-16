import cv2
import os
import numpy as np

input_folder = "/home/panque/repos/IA/Eigenface/animals/PetImages/Turtle"
output_folder = "/home/panque/repos/IA/Eigenface/animals/PetImagesx64/Turtlex64"
target_size = 64   # final image: 256x256

os.makedirs(output_folder, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

def resize_with_padding(img, target_size):

    old_h, old_w = img.shape[:2]
    scale = target_size / max(old_h, old_w)

    # Resize while keeping aspect ratio
    new_w = int(old_w * scale)
    new_h = int(old_h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded square background
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Compute centered position
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    return padded_img

for file in os.listdir(input_folder):
    if file.lower().endswith(valid_ext):
        path = os.path.join(input_folder, file)

        img = cv2.imread(path)

        if img is None:
            print("Skipping invalid file:", file)
            continue

        # Convert BGR → RGB (optional but recommended for NN training)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and center
        final_img = resize_with_padding(img, target_size)

        # Convert back RGB → BGR for saving
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(output_folder, file), final_img)
        print("Processed:", file)

print("All images resized successfully!")
