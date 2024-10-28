from itertools import permutations
import numpy as np
import pandas as pd
import torch
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation, MaskedAffineAutoregressiveTransform
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Function to load landmarks from Excel file (Only x, y coordinates)
def load_landmarks(excel_path):
    df = pd.read_excel(excel_path)
    landmarks = df.filter(like='_x').join(df.filter(like='_y')).to_numpy()  
    return landmarks

# Function to create Normalizing Flow model
def create_normalizing_flow(input_dim):
    base_distribution = StandardNormal([input_dim])  
    transforms = []

    for _ in range(5):  
        transforms.append(RandomPermutation(features=input_dim))  
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=256))  
    
    transform = CompositeTransform(transforms) #compose all the transforms 

# Function to load the trained model from the pth file we got from the previous training in NF_Train.py
def load_model(model_path, input_dim):
    flow = create_normalizing_flow(input_dim)  # Create the new flow model using the function above
    flow.load_state_dict(torch.load(model_path),strict=False)  
    flow.eval()  
    return flow

# Function to compute log probabilities for participant data
def compute_probabilities(flow, data_tensor):
    with torch.no_grad():  
        log_probabilities = flow.log_prob(data_tensor)  
    return log_probabilities

# Function to compute the mean of the distribution learned for Participant A
def compute_mean_of_distribution(flow_a, data_tensor):
    with torch.no_grad():
        mean_a = torch.mean(flow_a.log_prob(data_tensor))  
    return mean_a

# Function to compute norms based on the mean of Participant A's distribution
def compute_norms_relative_to_mean(flow_a, data_tensor, mean_a):
    with torch.no_grad():
        log_probabilities = flow_a.log_prob(data_tensor)
        norms = torch.abs(log_probabilities - mean_a)
    return norms

# Function to save probabilities and norms to Excel
def save_results_to_excel(probabilities, norms, participant_name_a, participant_name_b, output_folder):
    results_df = pd.DataFrame({
        "Log Probability": probabilities.numpy(),
        "Norm": norms.numpy()
    })
    output_path = os.path.join(output_folder, f"{participant_name_a}_vs_{participant_name_b}_results.xlsx")
    results_df.to_excel(output_path, index=False)
    print(f"Results saved for {participant_name_a} vs {participant_name_b} at {output_path}")

# Function to plot norms and save a graph for each permutation
def plot_norms(norms_a, other_norms, participant_name_a, other_participant_names, graphs_folder):
    plt.figure(figsize=(15, 8))  
    plt.hist(norms_a.numpy(), bins=30, alpha=0.5, color='blue', edgecolor='black', label=f'{participant_name_a} Norms')

    # this color map is used to get a unique color for each participant when we plot their norms in one graph
    colormap = cm.get_cmap('tab20', len(other_norms))  

    for idx, norms in enumerate(other_norms):
        color = colormap(idx)  
        plt.hist(norms.numpy(), bins=30, alpha=0.5, color=color, edgecolor='black', label=f'{other_participant_names[idx]} Norms')

    plt.title(f'Norms Distribution for {participant_name_a} and others')
    plt.xlabel('Norms')
    plt.ylabel('Count')
    plt.legend(fontsize='small', loc='upper right') 
    plt.grid(axis='y', alpha=0.75)
    plt_path = os.path.join(graphs_folder, f"{participant_name_a}_vs_others_norms_distribution.png")
    plt.savefig(plt_path, dpi=300) 
    plt.close()
    print(f"Main graph saved for {participant_name_a} vs others at {plt_path}")

    # for more visually appealing subplots, we can plot each participant's norms in a separate subplot
    num_subplots = len(other_norms)
    cols = 5  # Number of columns in the subplot grid
    rows = (num_subplots // cols) + (num_subplots % cols > 0)  # Calculate the number of rows needed
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 6, rows * 4))  

    axs = axs.flatten() if num_subplots > 1 else [axs]

    for idx, norms in enumerate(other_norms):
        color = colormap(idx)  
        axs[idx].hist(norms_a.numpy(), bins=30, alpha=0.5, color='blue', edgecolor='black', label=f'{participant_name_a} Norms')
        axs[idx].hist(norms.numpy(), bins=30, alpha=0.5, color=color, edgecolor='black', label=f'{other_participant_names[idx]} Norms')
        axs[idx].set_title(f'Comparison: {participant_name_a} vs {other_participant_names[idx]}')
        axs[idx].set_xlabel('Norms')
        axs[idx].set_ylabel('Count')
        axs[idx].legend(fontsize='small', loc='upper right')
        axs[idx].grid(axis='y', alpha=0.75)

    for idx in range(len(other_norms), len(axs)):
        fig.delaxes(axs[idx])

    fig.tight_layout()
    plt_path = os.path.join(graphs_folder, f"{participant_name_a}_subplots_vs_others.png")
    plt.savefig(plt_path, dpi=300)  # Save with higher resolution for better clarity
    plt.close()
    print(f"Subplot graph saved for {participant_name_a} at {plt_path}")

# Main logic: Loop through all participants as the baseline'A', and compute results for others
def compute_for_all_participants(participants, excel_folder, model_folder, output_folder, graphs_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(graphs_folder, exist_ok=True)
    
    # Loop over each participant as the base (A) and calculate norms for others
    for participant_a in participants:
        print(f"Processing {participant_a} as base (A)...")
        data_a = load_landmarks(f'{excel_folder}/{participant_a}_general_dataset_landmarks.xlsx')
        input_dim = data_a.shape[1]
        data_tensor_a = torch.tensor(data_a, dtype=torch.float32)

        model_path = os.path.join(model_folder, f'{participant_a}_NF_model.pth')
        flow_a = load_model(model_path, input_dim)
        mean_a = compute_mean_of_distribution(flow_a, data_tensor_a)

        other_participants = [p for p in participants if p != participant_a]  
        other_norms = []
        other_names = []
        
        for participant_b in other_participants:
            print(f"Computing for {participant_b} based on {participant_a}'s model...")
            data_b = load_landmarks(f'{excel_folder}/{participant_b}_general_dataset_landmarks.xlsx')
            data_tensor_b = torch.tensor(data_b, dtype=torch.float32)
            
            norms_b = compute_norms_relative_to_mean(flow_a, data_tensor_b, mean_a)
            other_norms.append(norms_b)
            other_names.append(participant_b)

            # Save individual results for each pair
            probabilities_b = compute_probabilities(flow_a, data_tensor_b)
            save_results_to_excel(probabilities_b, norms_b, participant_a, participant_b, output_folder)

        # Plot all norms in one graph for the current participant A
        norms_a = compute_norms_relative_to_mean(flow_a, data_tensor_a, mean_a)
        plot_norms(norms_a, other_norms, participant_a, other_names, graphs_folder)

# List of participants as exactly named in the Excel files
# each one will be considered as the base (A) and compared to others in the function above
participants = [
    '81651276_65_right_control_boring_2', '81651286_65_right_control_boring_2',
    '81661296_66_right', '81661306_66_right_control_boring_4',
    '81671316_67_right_control_boring_3', '81671326_67_left_control_boring_3',
    '81681336_68_right_control_boring_4', '81681346_68_right_control_boring_4',
    '81691356_69_right_control_boring_1', '81691366_69_right_control_boring_1',
    '81701376_70_right_control_boring_3', '81701386_70_right_control_boring_2',
    '81711396_71_right_control_boring_3', '81711406_71_right_control_boring_4',
    '81721416_72_right_control_boring_2', '81721426_72_right_control_boring_1',
    '81741456_74_right_control_boring_3', '81741466_74_right_control_boring_1',
    '81751476_75_right_control_boring_1', '81751486_75_right_control_boring_3',
    '81761496_76_right_control_boring_2', '81761506_76_right',
    '81771516_77_right_control_boring_3', '81771526_77_right_control_boring_2',
    '81791556_79_right_control_boring_1', '81791566_79_right_control_boring_3',
    '81801576_80_right_control_boring_3', '81801586_80_right_control_boring_4',
    '81821616_82_right_control_boring_1', '81821626_82_right_control_boring_1',
    '81831636_83_right_control_boring_1', '81831646_83_right_control_boring_3'
]


# Paths to the folders containing the Excel files, trained models, and output results
excel_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_NF'
model_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/NF_Models'
output_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/CompareResults'
graphs_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/CompareResults'

# Run the computation for all participants
compute_for_all_participants(participants, excel_folder, model_folder, output_folder, graphs_folder)
