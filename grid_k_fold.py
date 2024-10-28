import pandas as pd
import os
from PIL import Image
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import loguniform
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import uniform
import time

# First, we load the celebA dataset
num_images_to_process = 200000
print_every_image=5000
dataset_path = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/CelebA_dataset/list_attr_celeba.csv'  # Update with the dataset path
df = pd.read_csv(dataset_path)
df = df.head(num_images_to_process)

folder_path = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/CelebA_dataset/img_align_celeba/img_align_celeba'
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f).replace('\\', '/'))]
image_files.sort()

# here we define the target blendshapes
target_blendshapes = [
    'eyeSquintLeft', 
    'eyeSquintRight', 
    'mouthPressLeft', 
    'mouthPressRight', 
    'mouthSmileLeft', 
    'mouthSmileRight'
]


base_options = python.BaseOptions(model_asset_path='C:/Users/OdayA/Desktop/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def extract_blendshapes(image):
    image_np = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    results = detector.detect(mp_image)
    if results.face_landmarks:
        blendshape_values = [blendshape.score for blendshape in results.face_blendshapes[0] if blendshape.category_name in target_blendshapes]
        return blendshape_values
    return None

# Function to process a single image
def process_image(image_file):
    global count, img_num, failed_indices
    img_num += 1
    if(img_num % print_every_image ==0):
        print(f"Processing Image number:{img_num}")
    image_path = os.path.join(folder_path, image_file).replace('\\', '/')
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None

    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((128, 128))
    except Exception as e:
        print(f"Failed to load image: {image_path} with error: {e}")
        return None
    
    blendshapes = extract_blendshapes(image)
    if blendshapes is not None:
        return blendshapes
    else:
        count += 1
        failed_indices.append(img_num)
        return None


count = 0
img_num = 0
failed_indices = []

# now we want to process images in parallel to save time
blendshapes_data = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(process_image, image_file): image_file for image_file in image_files[:num_images_to_process]}
    for future in as_completed(futures):
        result = future.result()
        if result:
            blendshapes_data.append(result)

all_blendshapes_data = np.array(blendshapes_data)

print(f"Processed {len(all_blendshapes_data)} images in {time.time() - start_time} seconds.")
print(count)
df = df.drop(failed_indices).reset_index(drop=True)
print(f"Number of rows in the DataFrame: {df.shape[0]}")

print(f"Blendshapes array shape: {all_blendshapes_data.shape}")

if all_blendshapes_data.size == 0:
    print("No blendshapes data extracted.")

# Load the true labels
true_labels = df['Smiling'].values  
true_labels = np.where(true_labels == -1, 0, true_labels)


X = all_blendshapes_data
y = true_labels

# Define parameter grid using loguniform
param_dist = {
    'thresholds': [
        [
            loguniform(0.2, 1).rvs(),  # eyeSquintLeft
            loguniform(0.2, 1).rvs(),  # eyeSquintRight
            loguniform(0.001, 0.1).rvs(),  # mouthPressLeft
            loguniform(0.001, 0.1).rvs(),  # mouthPressRight
            loguniform(0.2, 0.6).rvs(),  # mouthSmileLeft
            loguniform(0.2, 0.6).rvs()   # mouthSmileRight
        ]
        for _ in range(1000)  # Generating 1000 sets of thresholds
    ]
}

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, thresholds=None):
        self.thresholds = thresholds

    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.all(X > self.thresholds, axis=1).astype(int)

print("Randomized Search Started")
random_search = RandomizedSearchCV(
    estimator=ThresholdClassifier(),
    param_distributions=param_dist,
    scoring=make_scorer(f1_score),
    cv=5,
    n_iter=1000,
    random_state=42,
    refit=True,
    n_jobs=-1
)
random_search.fit(X, y)

best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best F1 Score:", best_score)
