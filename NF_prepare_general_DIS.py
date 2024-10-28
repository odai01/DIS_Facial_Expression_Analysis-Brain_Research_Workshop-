import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Number of landmarks is 478(468 primary and additional 10 landmarks)
# each has 3 coordinates (x, y, z) but we are only interested in x and y
num_landmarks = 478

# Function to extract landmark data as a NumPy array (only x and y coordinates)
def extract_landmark_data(face_landmarks):
    landmarks = np.empty((num_landmarks * 2,))  # Adjust size for x, y only
    for idx, landmark in enumerate(face_landmarks):
        landmarks[idx * 2] = landmark.x
        landmarks[idx * 2 + 1] = landmark.y
    return landmarks


# Function to process the video and extract landmarks for each participant
def process_video(participant_name, video_path, excel_folder):
    # Initialize the MediaPipe FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='C:/Users/OdayA/Desktop/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # Video capture setup
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    processed_frames = 0
    max_frames = 2000  # maximum number of frames to process

    # Initialize a list to accumulate data for all frames
    all_frame_data = []

    while cap.isOpened() and processed_frames < max_frames:
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            landmark_array = extract_landmark_data(face_landmarks)
            
            # herer we ensure the array has the correct size of 956(478*2)
            assert landmark_array.size == num_landmarks * 2, "Landmark array size is incorrect."

            # Calculate the timestamp in seconds
            timestamp = frame_count / frame_rate
            # Combine landmark data with metadata (timestamp and frame number) to visualize using PCA
            frame_data = np.concatenate((landmark_array, [timestamp, frame_count]))
            all_frame_data.append(frame_data)

        processed_frames += 1
        frame_count += 1

    all_frame_data = np.array(all_frame_data)

    landmark_columns = []
    for i in range(num_landmarks):
        landmark_columns.append(f"landmark_{i}_x")
        landmark_columns.append(f"landmark_{i}_y")
    column_names = landmark_columns + ["timestamp", "frame_number"]

    df = pd.DataFrame(all_frame_data, columns=column_names)

    df.to_excel(f'{excel_folder}/{participant_name}_general_dataset_landmarks.xlsx', index=False)

    cap.release()

    return all_frame_data


# Function to visualize landmark distribution using PCA and save the plot
def visualize_landmarks(landmark_data, participant_name, output_folder):
    pca = PCA(n_components=2) # 2 components for 2D visualization
    reduced_data = pca.fit_transform(landmark_data[:, :-2])  # Exclude timestamp and frame number
    
    timestamps = landmark_data[:, -2]  # We will use timestamp as the color scale
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=timestamps, cmap='viridis', alpha=0.5)

    plt.colorbar(label='Timestamp (seconds)')
    plt.title(f'Landmark Distribution for {participant_name}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # Save the plot to a file
    output_path = f'{output_folder}/{participant_name}_landmark_distribution.png'
    plt.savefig(output_path)
    plt.show()  
    plt.clf()


# List of participant names to process, exactly as they appear in the video file names
participant_names = [
    '81771516_77_right_control_interesting_2', '81771526_77_right_control_interesting_3',
    '81791556_79_right_control_interesting_4', '81791566_79_right_control_interesting_4',
    '81801576_80_right_control_interesting_2', '81801586_80_right_control_interesting_1',
    '81821616_82_right_control_interesting_2', '81821626_82_right_control_interesting_3',
    '81831636_83_right_control_interesting_3', '81831646_83_right_control_interesting_1'
]

# Loop through each participant and process their video, to prepare their general landmark dataset
for participant_name in participant_names:
    print(f"Processing participant: {participant_name}")
    #the video path is the path to the video file, excel folder is the path to the folder where the excel file will be saved
    video_path = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_Videos/{participant_name}.mp4'
    excel_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_NF'
    
    landmarks = process_video(participant_name, video_path, excel_folder)
    
    visualize_landmarks(landmarks, participant_name,excel_folder)
