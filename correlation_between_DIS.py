import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations

def compare_dis(participant_names):
    results = []

    # Loop over all pairs of participants
    for participant_name1, participant_name2 in combinations(participant_names, 2):
        # Paths to the files
        general_or_smile='smile'
        path_dis1 = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_Excel/{general_or_smile}_DIS_{participant_name1}.csv'
        path_dis2 = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_Excel/{general_or_smile}_DIS_{participant_name2}.csv'
        
        dis1 = pd.read_csv(path_dis1)
        dis2 = pd.read_csv(path_dis2)
        
        # Ensure the feature pairs match in both files
        assert dis1['Feature_Pair'].tolist() == dis2['Feature_Pair'].tolist(), f"Feature pairs do not match for participants {participant_name1} and {participant_name2}"
        
        # Calculate Pearson and Spearman correlations between the two participants DIS
        pearson_correlation = np.corrcoef(dis1['Correlation'], dis2['Correlation'])[0, 1]
        spearman_correlation, _ = spearmanr(dis1['Correlation'], dis2['Correlation'])
        
        # Store the results
        results.append({
            'Comparison': f'Correlation_DIS_{participant_name1}_{participant_name2}',
            'Pearson': pearson_correlation,
            'Spearman': spearman_correlation
        })
    
    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results to a single Excel file
    output_path = f'C:/Users/OdayA/Desktop/Dynamic_Identity_Signature/Results_Excel/{general_or_smile}_correlation_DIS_comparisons.xlsx'
    results_df.to_excel(output_path, index=False)
    print("All comparison results saved to Excel.")

# the participant_list should be the list of the participants that we want to compare the DIS between each pair of them
participant_list = ['81651286_65_right_standup_4', '81661306_66_right_standup_1', '81661296_66_right_standup_2','81671326_67_right_standup_2','81671316_67_right_standup_4',
                    '81681336_68_right_standup_1','81691366_69_right_standup_4','81701376_70_right_standup_2','81711396_71_right_standup_1',
                    '81711406_71_right_standup_2','81741456_74_right_standup_1','81751476_75_right_standup_2','81751486_75_right_standup_2',
                    '81761496_76_right_standup_1','81761506_76_right_standup_1','81771516_77_right_standup_1','81771526_77_right_standup_4',
                    '81801586_80_right_standup_3','81821616_82_right_standup_3','81821626_82_right_standup_2','81831636_83_right_standup_2']
compare_dis(participant_list)
