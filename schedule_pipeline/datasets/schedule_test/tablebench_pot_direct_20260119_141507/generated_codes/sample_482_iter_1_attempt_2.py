import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Identify the row with Liiga totals (last row)
totals_row = df.iloc[-1]
liiga_totals = {
    'G': totals_row['Regular season_1'],
    'A': totals_row['Regular season_2'],
    'Pts': totals_row['Regular season_3']
}

# Extract regular season data (excluding the last row)
season_data = df.iloc[:-1]
season_data = season_data[season_data['Season'].notna()]

# Compute deviations for each season in Regular season
deviations = []
for idx, row in season_data.iterrows():
    reg_g = row['Regular season_1']
    reg_a = row['Regular season_2']
    reg_pts = row['Regular season_3']
    
    # Calculate absolute deviations from totals
    g_dev = abs(reg_g - liiga_totals['G'])
    a_dev = abs(reg_a - liiga_totals['A'])
    pts_dev = abs(reg_pts - liiga_totals['Pts'])
    
    # Define threshold: if deviation is more than 10% of the total, consider it significant
    g_threshold = liiga_totals['G'] * 0.1
    a_threshold = liiga_totals['A'] * 0.1
    pts_threshold = liiga_totals['Pts'] * 0.1
    
    if g_dev > g_threshold or a_dev > a_threshold or pts_dev > pts_threshold:
        deviations.append({
            'Season': row['Season'],
            'G_dev': g_dev,
            'A_dev': a_dev,
            'Pts_dev': pts_dev
        })

# If deviations exist, list the seasons; otherwise, say none
if deviations:
    significant_seasons = [d['Season'] for d in deviations]
    print(f"Final Answer: {', '.join(significant_seasons)}")
else:
    print("Final Answer: none")