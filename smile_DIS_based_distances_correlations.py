import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def process_video(participant_name):
    # the same as the previous code, video path is the path to the video file on the pc
    # smile_seconds_path is the path to the excel file that contains the seconds where the participant smiled(which we got from the smile detection model)
    # output_excel_path is the path where the excel file that contains the DIS based correlation results will be saved
    video_path = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_Videos/{participant_name}.mp4'
    smile_seconds_path = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_Excel/smile_seconds_{participant_name}.xlsx'
    output_excel_path = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_Excel/smile_DIS_{participant_name}.csv'

    # Read smile detection data
    smile_seconds_df = pd.read_excel(smile_seconds_path)
    smile_seconds = smile_seconds_df['second'].tolist()

    # Create a set of seconds to process, including the second before and after each smile second
    # we included also the second before and after the smile second to make each smiling participant has at least 3 seconds to process
    seconds_to_process = set()
    for sec in smile_seconds:
        seconds_to_process.update([sec - 1, sec, sec + 1])
    print(seconds_to_process)
    cap = cv2.VideoCapture(video_path)

    distance_changes = {}

    # the 33 landmarks of the face we are interested in
    regions = {
        "mouth": [61, 291, 13, 14],
        "left_eye": [33, 133, 23, 159],
        "right_eye": [362, 263, 253, 258],
        "nose": [1, 2, 98, 327],
        "left_eyebrow": [63, 55],
        "right_eyebrow": [283, 285],
        "chin": [152, 365, 136, 194, 418],
        "left_cheek": [234, 192],
        "right_cheek": [454, 433],
        "head": [9, 10, 104, 333]
    }

    all_landmarks = [idx for indices in regions.values() for idx in indices]

    def euclidean_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    frame_count = 0
    fps = 25  # Frames per second
    sample_rate = fps

    prev_landmark_coords = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_second = frame_count // fps

        # Process the frames for seconds in the set
        if current_second in seconds_to_process:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    landmark_coords = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

                    if prev_landmark_coords is not None:
                        for i in range(len(all_landmarks)):
                            for j in range(i + 1, len(all_landmarks)):
                                idx1 = all_landmarks[i]
                                idx2 = all_landmarks[j]
                                dist_change = euclidean_distance(landmark_coords[idx1], landmark_coords[idx2]) - \
                                              euclidean_distance(prev_landmark_coords[idx1], prev_landmark_coords[idx2])
                                feature_key = f"dist_change_{idx1}_{idx2}"
                                
                                if feature_key not in distance_changes:
                                    distance_changes[feature_key] = []
                                distance_changes[feature_key].append(dist_change)

                    prev_landmark_coords = landmark_coords

        frame_count += 1

    cap.release()

    # Standardize the distances
    standardized_distance_changes = {}
    for key, values in distance_changes.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        standardized_values = [(val - mean_val) / std_val for val in values]
        standardized_distance_changes[key] = standardized_values

    # Calculate correlations between the standardized distance changes over time
    correlation_results = {}
    feature_keys = list(standardized_distance_changes.keys())

    for i in range(len(feature_keys)):
        for j in range(i + 1, len(feature_keys)):
            corr = np.corrcoef(standardized_distance_changes[feature_keys[i]], standardized_distance_changes[feature_keys[j]])[0, 1]
            correlation_results[f"{feature_keys[i]}_vs_{feature_keys[j]}"] = corr

    # Save correlation results to CSV
    correlation_df = pd.DataFrame(list(correlation_results.items()), columns=['Feature_Pair', 'Correlation'])
    correlation_df.to_csv(output_excel_path, index=False)
    print(f"DIS data for {participant_name} saved to Excel.")

# the last line should be the name of the participant we want to process, exactly as in the video file name
process_video('81831636_83_right_standup_2')
