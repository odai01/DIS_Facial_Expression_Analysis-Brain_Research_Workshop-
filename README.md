# Smile Detection and Dynamic Identity Signature (DIS) based Correlations

This section documents the code files used for extracting blendshapes, detecting smiles, constructing the Dynamic Identity Signature (DIS) based on correlations, and comparing DIS metrics between partners and strangers. The scripts focus on building smile-based DIS representations and comparing them to study similarity in dynamic facial expressions.

---

## Files and Functionality

### 1. **grid_k_fold.py**
This script performs a grid search to determine optimal thresholds for smile detection based on a blendshape-based approach.

- **Process:**  
  - Uses the CelebA dataset with labeled images. Images are processed in parallel, extracting blendshape scores for key facial expressions.
  - The script applies `RandomizedSearchCV` to identify optimal thresholds for smile-indicating blendshapes (e.g., eye squint, mouth press).
- **Outputs:**  
  - Best parameter thresholds for each target blendshape, enhancing smile detection accuracy.
- **Note:**  
  - Ensure paths to the dataset are correctly updated in the code (lines with `dataset_path` and `folder_path`).

### 2. **detect_smiles.py**
This script detects smiles in video frames using blendshape threshold values derived from `grid_k_fold.py`.

- **Process:**  
  - Loads participant video files and extracts blendshape scores per frame.
  - Identifies smile instances based on blendshape thresholds and logs smiling frames and seconds.
- **Outputs:**  
  - Excel files containing smile detection results for each frame and second.
- **Note:**  
  - Update `video_path` and `excel_folder` paths to point to the correct video and results directories.

### 3. **smile_DIS_based_distances_correlations.py**
This script constructs the Dynamic Identity Signature (DIS) by calculating correlations between changes in facial distances over time between key landmarks.

- **Process:**  
  - Processes each second containing smiles as detected in the prior step.
  - Calculates the changes in Euclidean distances between selected landmark pairs and standardizes these values.
  - Computes the correlations between standardized distance changes over time to form the DIS matrix.
- **Outputs:**  
  - An Excel file containing the DIS matrix for the participant, showing correlations for each feature pair.
- **Note:**  
  - Update `video_path`, `smile_seconds_path`, and `output_excel_path` to match the locations of input video files and output results.

### 4. **correlation_between_DIS.py**
This script compares DIS matrices between pairs of participants, quantifying similarity in their smile dynamics.

- **Process:**  
  - Calculates Pearson and Spearman correlations between DIS matrices for participant pairs to evaluate both linear and monotonic relationships.
- **Outputs:**  
  - An Excel file with correlation scores (Pearson and Spearman) for each pair, indicating the degree of similarity in their smile-based DIS.
- **Note:**  
  - Update file paths to DIS matrices (`path_dis1` and `path_dis2`) in the code as needed.

---

These files collectively support the smile-based DIS analysis by identifying optimal smile detection thresholds, detecting smiles in video frames, constructing DIS matrices, and performing statistical comparisons to assess similarity between participants.
