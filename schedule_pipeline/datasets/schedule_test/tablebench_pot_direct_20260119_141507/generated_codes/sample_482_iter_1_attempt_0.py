import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop the last row (totals) since it's not a season
df = df.iloc[:-1]

# Extract relevant columns
regular_cols = ['GP', 'G', 'A', 'Pts']
playoff_cols = ['GP', 'G', 'A', 'Pts']

# Extract the total Liiga values (last row)
total_row = df.iloc[-1]
total_gp, total_g, total_a, total_pts = total_row['GP'], total_row['G'], total_row['A'], total_row['Pts']

# Create a list to store deviations
deviations = []

# Loop through each season row
for idx, row in df.iterrows():
    team = row['Team']
    season = row['Season']
    
    # Skip rows without regular season data
    if pd.isna(row['GP']) or row['GP'] == '—':
        continue
        
    reg_gp, reg_g, reg_a, reg_pts = row['GP'], row['G'], row['A'], row['Pts']
    
    # Compute ratios
    gp_ratio = reg_gp / total_gp if total_gp > 0 else 0
    g_ratio = reg_g / total_g if total_g > 0 else 0
    a_ratio = reg_a / total_a if total_a > 0 else 0
    pts_ratio = reg_pts / total_pts if total_pts > 0 else 0
    
    # Compute standard deviation of ratios across all seasons (only for Liiga teams)
    # We'll assume only HIFK and JYP are relevant (Liiga teams)
    # For simplicity, we check if any ratio is more than 2 standard deviations from mean
    # We'll compute mean and std from all valid Liiga teams' ratios
    
    # Only consider rows where team is in Liiga (e.g., HIFK, JYP)
    if team in ['HIFK', 'JYP Jyväskylä']:
        # Compute mean and std of ratios for this team
        # We'll use the ratios from the current season
        ratios = [gp_ratio, g_ratio, a_ratio, pts_ratio]
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        # Flag if any ratio is more than 2 std from mean
        if std_ratio > 0:
            deviation = np.max([abs(r - mean_ratio) for r in ratios]) / std_ratio > 2
            if deviation:
                deviations.append((season, team))

# Output seasons with significant deviation
if deviations:
    print(f"Final Answer: {', '.join([f'{s} ({t})' for s, t in deviations])}")
else:
    print("Final Answer: None")