# Dynamic Identity Signatures (DIS) in Facial Expressions: Correlations and Normalizing Flow Analysis

This repository includes code and documentation for two main approaches in analyzing Dynamic Identity Signatures (DIS) in facial expressions: (1) **Smile-Based DIS Correlations**, where smile detection and facial distance correlations are used to build participant-specific DIS matrices, and (2) **DIS-Based Normalizing Flows (NF)**, where high-dimensional DIS data is modeled through Normalizing Flow algorithms to assess pairwise and group-based similarities.

Each approach is organized as follows:

- **Section 1: Smile Detection and DIS-Based Correlations** – focused on extracting blendshapes, detecting smiles, calculating DIS matrices from smile-based correlations, and comparing participant DIS data for similarity.
- **Section 2: DIS-Based Normalizing Flows** – focused on training NF models with facial expression data, calculating log probabilities and norms, and analyzing aggregated similarity results for partner and stranger pairs.

---

## Section 1: Smile Detection and DIS Based Correlations

This section includes the scripts for extracting blendshapes, detecting smiles, creating DIS matrices based on correlations between landmarks, and comparing these matrices to assess similarities.

---

### Files and Functionality

#### 1. **grid_k_fold.py**
This script performs a grid search to find optimal thresholds for smile detection based on blendshape scores.

- **Process:**  
  - Uses the CelebA dataset with labeled images, processes them in parallel, and extracts blendshape scores.
  - Uses `RandomizedSearchCV` to identify thresholds for smile-indicating blendshapes.
- **Outputs:**  
  - Best parameter thresholds for blendshapes to enhance smile detection.
- **Note:**  
  - Update paths to the dataset (`dataset_path` and `folder_path`) as needed.

#### 2. **detect_smiles.py**
Detects smiles in video frames using blendshape thresholds derived from `grid_k_fold.py`.

- **Process:**  
  - Processes participant video files to detect smile frames based on blendshape thresholds.
- **Outputs:**  
  - Excel files containing smile detection results for each frame and second.
- **Note:**  
  - Update `video_path` and `excel_folder` paths to point to video and results directories.

#### 3. **smile_DIS_based_distances_correlations.py**
Constructs the Dynamic Identity Signature (DIS) by calculating correlations between changing facial distances over time.

- **Process:**  
  - Processes seconds with smiles, calculates changes in Euclidean distances between landmark pairs, and standardizes these values.
  - Generates a DIS matrix representing correlations for each feature pair.
- **Outputs:**  
  - An Excel file with the DIS matrix showing correlations for each feature pair.
- **Note:**  
  - Update paths (`video_path`, `smile_seconds_path`, and `output_excel_path`) to match input video files and output results.

#### 4. **correlation_between_DIS.py**
Compares DIS matrices between participant pairs, quantifying similarity in their smile dynamics.

- **Process:**  
  - Calculates Pearson and Spearman correlations between DIS matrices for participant pairs.
- **Outputs:**  
  - Excel file with Pearson and Spearman correlation scores for each pair.
- **Note:**  
  - Update paths to DIS matrices (`path_dis1` and `path_dis2`) as needed.

---

## Section 2: DIS Based Normalizing Flows (NF)

This section documents the code files for training and evaluating DIS-based Normalizing Flows (NF) models to analyze dynamic identity signatures in high-dimensional data.

---

### Files and Functionality

#### 1. **NF_prepare_general_DIS.py**
Prepares general landmark datasets by extracting and visualizing facial landmarks for each participant.

- **Process:**  
  - Extracts and stores x and y coordinates of 478 facial landmarks.
  - Applies PCA to visualize landmark distribution over time.
- **Outputs:**  
  - Excel files with landmark coordinates and timestamps.
  - PCA visualizations of landmark distribution.
- **Note:**  
  - Update `video_path` and `excel_folder` paths as needed.

#### 2. **NF_train.py**
Trains NF models on participant data to capture unique signature patterns.

- **Process:**  
  - Loads landmark data and trains an NF model with specified layers and learning rate.
  - Saves trained NF model and loss plot.
- **Outputs:**  
  - `.pth` model files for each participant.
  - Loss plot for visualizing model convergence.
- **Note:**  
  - Update paths to `excel_folder`, `output_folder`, and `loss_folder` for data and models.

#### 3. **NF_pairwise_norms.py**
Computes pairwise comparisons by applying each participant’s NF model to data from other participants.

- **Process:**  
  - Calculates norms and log probabilities, saving results and generating distribution plots.
- **Outputs:**  
  - Excel files for pairwise comparisons.
  - Norm distribution graphs comparing participants.
- **Note:**  
  - Update `excel_folder`, `model_folder`, and `graphs_folder` paths for loading data and saving results.

#### 4. **NF_combined_norms.py**
Aggregates norms from all comparisons to provide comprehensive statistical results.

- **Process:**  
  - Combines norms for partners and strangers, conducting statistical tests (t-tests and Mann-Whitney U).
  - Generates histograms for visual comparisons.
- **Outputs:**  
  - Aggregated histograms and statistical test results.
- **Note:**  
  - Update paths to folders for input and output files as required.

---
Together, these files provide a comprehensive framework for analyzing facial expression dynamics and DIS similarity in both smile-based correlations and high-dimensional data modeling using Normalizing Flows.
