import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from deepface import DeepFace  
import sys
import shutil
from tqdm import tqdm

def print_help():
    help_text = """
Usage: python main.py <image_path> <group_dir> <output_dir>

Arguments:
  image_path     Path to the input image containing people to be detected.
  group_dir      Directory containing group images for face comparison.
  output_dir     Directory where cropped images and matched group images will be saved.

Example:
  python main.py /path/to/input.jpg /path/to/group_images /path/to/output_dir

Description:
  - The tool first detects people in the input image using YOLO.
  - It crops each person from the image and saves the cropped images in the specified output directory.
  - Then, for each cropped person image, it compares it with images in the group directory using the DeepFace library (Facenet model).
  - If a match is found, the matching group image is copied to the corresponding subdirectory in the output directory.

Flags:
  -h, --help     Show this help message and exit.
"""
    print(help_text)


image_path = sys.argv[1:2]
group_dir = sys.argv[2:3]
output_dir = sys.argv[3:]

if len(sys.argv) < 4:
    print("Error: Missing required arguments.")
    print_help()
    sys.exit(1)



if '--help' in sys.argv or '-h' in sys.argv:
    print_help()
    sys.exit()

# If not help, then run the main functionality
model = YOLO('yolov8n.pt')
img = cv2.imread(image_path[0])
results = model(img)
detections = results[0].boxes
cropped_images = []
for box in detections:
    if box.cls==0:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 
        cropped_img = img[y1:y2, x1:x2] 
        cropped_images.append(cropped_img)

os.makedirs(output_dir[0], exist_ok=True)

for i, crop in enumerate(cropped_images):
    cv2.imwrite(os.path.join(output_dir[0], f'person_{i+1}.jpg'), crop)

source1 = output_dir
count = 0

for image in tqdm(os.listdir(source1[0]), desc="Processing source images"):
    pp = os.path.join(source1[0], image)
    output_person_dir = os.path.join(source1[0] + '/output/', str(count))
    os.makedirs(output_person_dir, exist_ok=True)
    shutil.copy(pp, output_person_dir)

    for group_image in tqdm(os.listdir(group_dir[0]), desc="Comparing with group images", leave=False):
        kk = os.path.join(group_dir[0], group_image)

        if not os.path.exists(pp):
            print(f"Source image {pp} does not exist.")
            continue
        if not os.path.exists(kk):
            print(f"Group image {kk} does not exist.")
            continue

       # result = DeepFace.verify(img1_path=pp, img2_path=kk, model_name='Facenet')         
        result = DeepFace.verify(img1_path=pp, img2_path=kk, model_name='Facenet',enforce_detection=False)
 
        if result['verified']:
            shutil.copy(kk, output_person_dir)

    count += 1

