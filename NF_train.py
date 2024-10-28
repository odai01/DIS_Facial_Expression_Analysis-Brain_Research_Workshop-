import numpy as np
import pandas as pd
import torch
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Function to load landmarks from the excel file produced by the previous script NF_prepare_general_DIS.py
def load_landmarks(excel_path):
    df = pd.read_excel(excel_path)
    landmarks = df.drop(columns=['timestamp', 'frame_number']).to_numpy()
    return landmarks

# Function to create Normalizing Flow model with more layers and larger flow size
def create_normalizing_flow(input_dim):
    base_distribution = StandardNormal([input_dim])
    transforms = []

    # creating the normalizing flow with 8 layers and 256 hidden features
    for _ in range(8): 
        transforms.append(RandomPermutation(features=input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=256)) 
    
    transform = CompositeTransform(transforms)
    return Flow(transform, base_distribution)

# Function to train Normalizing Flow on participant data
def train_normalizing_flow(flow, data_tensor, epochs=800, learning_rate=1e-4):
    print(f"Shape of data_tensor: {data_tensor.shape}")
    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    
    # we store the loss values to plot them later
    loss_values = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = -flow.log_prob(data_tensor).mean()
        loss.backward()

        # Apply gradient clipping to avoid large updates, and thus to avoid getting away from the optimal solution
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_values.append(loss.item()) 
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

    return loss_values  

# we save the model(nf for each participant) as a pth file
def save_model_and_distribution(flow, participant, output_folder):
    model_path = os.path.join(output_folder, f'{participant}_NF_model.pth')
    print(f"Model path: {model_path}")

    try:
        torch.save(flow.state_dict(), model_path)
        print(f"Model saved for {participant}")
    except Exception as e:
        print(f"Error saving model for {participant}: {e}")

# Function to plot and save loss values
def plot_loss(loss_values, participant, loss_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title(f'Loss over Epochs for {participant}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    loss_path = os.path.join(loss_folder, f'{participant}_loss_plot.png')
    plt.savefig(loss_path)
    plt.close()  

# the main function that trains the NF model for each participant and saves the model and loss plot
def train_and_save(participant_names, excel_folder, output_folder, loss_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(loss_folder, exist_ok=True)  # Create loss folder if it doesn't exist

    for participant in participant_names:
        print(f"Processing participant {participant}...")
        data = load_landmarks(f'{excel_folder}/{participant}_general_dataset_landmarks.xlsx')
        input_dim = data.shape[1]
        flow = create_normalizing_flow(input_dim)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        print(f"Training Normalizing Flow for {participant}...")
        loss_values = train_normalizing_flow(flow, data_tensor)
        print(f"Training completed for {participant}.")
        
        print(f"Saving model and distribution for {participant}...")
        save_model_and_distribution(flow, participant, output_folder)
        
        print(f"Plotting loss for {participant}...")
        plot_loss(loss_values, participant, loss_folder)  # Plot and save loss values
        print(f"Saved model and loss plot for {participant}.")

# List of participant names exactly as they appear in the video file names and thus in the excel files
participant_names = [
    '81771516_77_right_control_interesting_2', '81771526_77_right_control_interesting_3',
    '81791556_79_right_control_interesting_4', '81791566_79_right_control_interesting_4',
    '81801576_80_right_control_interesting_2', '81801586_80_right_control_interesting_1',
    '81821616_82_right_control_interesting_2', '81821626_82_right_control_interesting_3',
    '81831636_83_right_control_interesting_3', '81831646_83_right_control_interesting_1'
]
# Folders paths for data and model saving
excel_folder = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_NF'
output_folder = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/NF_Models'
loss_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/LossGraphs'

# Run the training and saving process
train_and_save(participant_names, excel_folder, output_folder, loss_folder)
