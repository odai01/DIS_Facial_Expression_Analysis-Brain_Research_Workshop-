import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation, MaskedAffineAutoregressiveTransform
from sklearn.utils import resample
from scipy.stats import mannwhitneyu

def load_landmarks(excel_path):
    df = pd.read_excel(excel_path)
    landmarks = df.filter(like='_x').join(df.filter(like='_y')).to_numpy()  # Keep only x, y columns
    return landmarks

def create_normalizing_flow(input_dim):
    base_distribution = StandardNormal([input_dim])
    transforms = []
    for _ in range(5):
        transforms.append(RandomPermutation(input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(input_dim, 256))
    transform = CompositeTransform(transforms)
    return Flow(transform, base_distribution)

def load_model(model_path, input_dim):
    flow = create_normalizing_flow(input_dim)
    flow.load_state_dict(torch.load(model_path), strict=False)
    flow.eval()
    return flow

# Calculate both log probabilities and norms based on log probabilities relative to their mean
def compute_probabilities_and_norms(flow, data_tensor):
    with torch.no_grad():
        log_probabilities = flow.log_prob(data_tensor)
        mean_log_prob = log_probabilities.mean()
        norms = torch.abs(log_probabilities - mean_log_prob)  
    return log_probabilities.numpy(), norms.numpy()

# Read norms from the comparison Excel file if it exists(since we already saved most of them in NF_pairwise_norms.py), or calculate if missing
def get_norms(participant_a, participant_b, compare_folder, results_nf_folder, model_folder, recalculate_missing=True):
    excel_path = f'{compare_folder}/{participant_a}_vs_{participant_b}_results.xlsx'
    
    if os.path.exists(excel_path):  # Read from file if it exists
        print(f"Reading existing file: {excel_path}")
        return pd.read_excel(excel_path)['Norm'].to_numpy()
    
    elif recalculate_missing:  # Calculate if missing
        print(f"File missing. Calculating norms between {participant_a} and {participant_b}...")
        model_path = f'{model_folder}/{participant_a}_NF_model.pth'
        landmarks_b_path = f'{results_nf_folder}/{participant_b}_general_dataset_landmarks.xlsx'
        
        data_b = torch.tensor(load_landmarks(landmarks_b_path), dtype=torch.float32)
        flow = load_model(model_path, data_b.shape[1])
        
        _, norms = compute_probabilities_and_norms(flow, data_b)
        return norms

# Main function to calculate and plot the histograms
def plot_combined_histograms(participants, partner_mapping, unrelated_mapping, compare_folder, results_nf_folder, model_folder, output_folder, recalculate_missing=True):
    base_norms_all = []
    partner_norms_all = []
    stranger_norms_all = []

    for participant in participants:
        landmarks_path = f'{results_nf_folder}/{participant}_general_dataset_landmarks.xlsx'
        model_path = f'{model_folder}/{participant}_NF_model.pth'
        data = torch.tensor(load_landmarks(landmarks_path), dtype=torch.float32)
        flow = load_model(model_path, data.shape[1])

        # Calculate the base norms
        _, base_norms = compute_probabilities_and_norms(flow, data)
        base_norms_all.extend(base_norms)

        # Load partner norms
        if(partner_mapping[participant] != None):
            partner_name = partner_mapping[participant]
            partner_norms = get_norms(participant, partner_name, compare_folder, results_nf_folder, model_folder, recalculate_missing)
            partner_norms_all.extend(partner_norms)

        # Load stranger norms
        for stranger_name in unrelated_mapping[participant]:
            stranger_norms = get_norms(participant, stranger_name, compare_folder, results_nf_folder, model_folder, recalculate_missing)
            stranger_norms_all.extend(stranger_norms)

    # Plot combined histograms for base, partner, and stranger norms
    plt.figure(figsize=(12, 8))
    plt.hist(base_norms_all, bins=30, alpha=1, color='blue', edgecolor='black', label='Base Norms')
    plt.hist(partner_norms_all, bins=30, alpha=0.5, color='green', edgecolor='black', label='Partner Norms')
    plt.hist(stranger_norms_all, bins=30, alpha=0.5, color='red', edgecolor='black', label='Stranger Norms')
    plt.title('Combined Norms: Base, Partner, Stranger')
    plt.xlabel('Norms')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    # performing the t-tests
    t_partner, p_partner = ttest_ind(base_norms_all, partner_norms_all, equal_var=False)
    t_unrelated, p_unrelated = ttest_ind(base_norms_all, stranger_norms_all, equal_var=False)

    # Adjusting the lengths of arrays to match for comparison for the third t test
    min_length = min(len(base_norms_all), len(partner_norms_all), len(stranger_norms_all))
    base_norms_all = resample(base_norms_all, n_samples=min_length, replace=False, random_state=42)
    partner_norms_all = resample(partner_norms_all, n_samples=min_length, replace=False, random_state=42)
    stranger_norms_all = resample(stranger_norms_all, n_samples=min_length, replace=False, random_state=42)
    base_partner_diff = np.abs(np.array(base_norms_all) - np.array(partner_norms_all))
    base_stranger_diff = np.abs(np.array(base_norms_all) - np.array(stranger_norms_all))
    t_diff, p_diff = ttest_ind(base_stranger_diff, base_partner_diff, alternative='greater',equal_var=False)
    # Perform Mann-Whitney U tests, as an additional non-parametric test
    mw_partner_stat, mw_partner_p = mannwhitneyu(base_norms_all, partner_norms_all, alternative='two-sided')
    mw_unrelated_stat, mw_unrelated_p = mannwhitneyu(base_norms_all, stranger_norms_all, alternative='two-sided')
    mw_diff_stat, mw_diff_p = mannwhitneyu(base_stranger_diff, base_partner_diff, alternative='greater')

    print(f'T-test Base vs Partner p-val and tval: {p_partner}, {t_partner}')
    print(f'T-test Base vs Unrelated p-val and tval: {p_unrelated}, {t_unrelated}')
    print(f'T-test One-sided diff test p-val and tval: {p_diff}, {t_diff}')

    print(f'Mann-Whitney U Base vs Partner p-val and U-stat: {mw_partner_p}, {mw_partner_stat}')
    print(f'Mann-Whitney U Base vs Unrelated p-val and U-stat: {mw_unrelated_p}, {mw_unrelated_stat}')
    print(f'Mann-Whitney U One-sided diff test p-val and U-stat: {mw_diff_p}, {mw_diff_stat}')
    # Add t-test results to the plot
    plt.figtext(0.5, 0.85, f'Base vs Partner p-val: {p_partner:.4f}', ha='center', color='blue')
    plt.figtext(0.5, 0.82, f'Base vs Unrelated p-val: {p_unrelated:.4f}', ha='center', color='blue')
    plt.figtext(0.5, 0.79, f'One-sided diff test p-val: {p_diff:.4f}', ha='center', color='blue')
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'combined_histograms_full.png'))
    
    # Save zoomed-in versions of the plot
    plt.xlim(0, 50000)  
    plt.savefig(os.path.join(output_folder, 'combined_histograms_zoomed1.png'))
    plt.xlim(0, 100000) 
    plt.savefig(os.path.join(output_folder, 'combined_histograms_zoomed2.png'))
    plt.xlim(0, 200000)  
    plt.savefig(os.path.join(output_folder, 'combined_histograms_zoomed3.png'))
    plt.show()


# participants list which each will be considere as base 
participants = [
    '81651276_65_right_control_interesting_3', '81651286_65_right_control_interesting_3',
    '81661296_66_right_control_interesting_1', '81661306_66_right_control_interesting_3',
    '81671316_67_right_control_interesting_2', '81671326_67_right_control_interesting_2',
    '81681336_68_right_control_interesting_2', '81681346_68_right_control_interesting_2',
    '81691356_69_right_control_interesting_4', '81691366_69_right_control_interesting_3',
    '81701376_70_right_control_interesting_4', '81701386_70_right_control_interesting_3',
    '81711396_71_right_control_interesting_4', '81711406_71_right_control_interesting_1',
    '81721416_72_right_control_interesting_1', '81721426_72_right_control_interesting_3',
    '81741456_74_right_control_interesting_4', '81741466_74_right_control_interesting_4',
    '81751476_75_right_control_interesting_4', '81751486_75_right_control_interesting_1',
    '81761496_76_right_control_interesting_3', '81761506_76_right_control_interesting_3',
    '81771516_77_right_control_interesting_2', '81771526_77_right_control_interesting_3',
    '81791556_79_right_control_interesting_4', '81791566_79_right_control_interesting_4',
    '81801576_80_right_control_interesting_2', '81801586_80_right_control_interesting_1',
    '81821616_82_right_control_interesting_2', '81821626_82_right_control_interesting_3',
    '81831636_83_right_control_interesting_3', '81831646_83_right_control_interesting_1'
]

# Partner mapping
partner_mapping = {
    '81651276_65_right_control_interesting_3': '81651286_65_right_control_interesting_3',
    '81651286_65_right_control_interesting_3': '81651276_65_right_control_interesting_3',

    '81661296_66_right_control_interesting_1': '81661306_66_right_control_interesting_3',
    '81661306_66_right_control_interesting_3': '81661296_66_right_control_interesting_1',

    '81671316_67_right_control_interesting_2': '81671326_67_right_control_interesting_2',
    '81671326_67_right_control_interesting_2': '81671316_67_right_control_interesting_2',

    '81681336_68_right_control_interesting_2': '81681346_68_right_control_interesting_2',
    '81681346_68_right_control_interesting_2': '81681336_68_right_control_interesting_2',

    '81691356_69_right_control_interesting_4': '81691366_69_right_control_interesting_3',
    '81691366_69_right_control_interesting_3': '81691356_69_right_control_interesting_4',

    '81701376_70_right_control_interesting_4': '81701386_70_right_control_interesting_3',
    '81701386_70_right_control_interesting_3': '81701376_70_right_control_interesting_4',

    '81711396_71_right_control_interesting_4': '81711406_71_right_control_interesting_1',
    '81711406_71_right_control_interesting_1': '81711396_71_right_control_interesting_4',

    '81721416_72_right_control_interesting_1': '81721426_72_right_control_interesting_3',
    '81721426_72_right_control_interesting_3': '81721416_72_right_control_interesting_1',

    '81741456_74_right_control_interesting_4': '81741466_74_right_control_interesting_4',
    '81741466_74_right_control_interesting_4': '81741456_74_right_control_interesting_4',

    '81751476_75_right_control_interesting_4': '81751486_75_right_control_interesting_1',
    '81751486_75_right_control_interesting_1': '81751476_75_right_control_interesting_4',

    '81761496_76_right_control_interesting_3': '81761506_76_right_control_interesting_3',
    '81761506_76_right_control_interesting_3': '81761496_76_right_control_interesting_3',

    '81771516_77_right_control_interesting_2': '81771526_77_right_control_interesting_3',
    '81771526_77_right_control_interesting_3': '81771516_77_right_control_interesting_2',

    '81791556_79_right_control_interesting_4': '81791566_79_right_control_interesting_4',
    '81791566_79_right_control_interesting_4': '81791556_79_right_control_interesting_4',

    '81801576_80_right_control_interesting_2': '81801586_80_right_control_interesting_1',
    '81801586_80_right_control_interesting_1': '81801576_80_right_control_interesting_2',

    '81821616_82_right_control_interesting_2': '81821626_82_right_control_interesting_3',
    '81821626_82_right_control_interesting_3': '81821616_82_right_control_interesting_2',

    '81831636_83_right_control_interesting_3': '81831646_83_right_control_interesting_1',
    '81831646_83_right_control_interesting_1': '81831636_83_right_control_interesting_3'
}

# Unrelated mapping
unrelated_mapping = {
    participant: [
        other_participant for other_participant in participants
        if other_participant != participant and other_participant != partner_mapping.get(participant)
    ]
    for participant in participants
}

# compare folder is the folder where the comparison results from the NF_pairwise_norms.py were saved
compare_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/CompareResults'
# results_nf_folder is the folder where the NF excel results were saved from the NF_train.py
results_nf_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_NF'
# model_folder is the folder where the NF models were saved from the NF_train.py
model_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/NF_Models'
# output_folder is the folder where the output histograms will be saved
output_folder = 'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/FinalResults'

plot_combined_histograms(participants, partner_mapping, unrelated_mapping, compare_folder, results_nf_folder, model_folder, output_folder)
