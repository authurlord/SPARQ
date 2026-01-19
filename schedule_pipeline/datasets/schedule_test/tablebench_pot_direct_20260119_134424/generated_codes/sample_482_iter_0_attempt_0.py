import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract career totals (last row)
career_total = df.iloc[-1]
career_gp = career_total['Regular season']
career_g = career_total['Regular season_2']
career_a = career_total['Regular season_3']
career_pts = career_total['Regular season_4']

# Calculate career average per game
career_avg_g = career_g / career_gp
career_avg_a = career_a / career_gp
career_avg_pts = career_pts / career_gp

# Initialize list to store deviating seasons
deviating_seasons = []

# Iterate through each season (excluding the total row)
for idx, row in df.iloc[:-1].iterrows():
    gp = row['Regular season']
    g = row['Regular season_2']
    a = row['Regular season_3']
    pts = row['Regular season_4']
    
    # Skip if GP is not numeric (e.g., '—')
    if gp == '—' or g == '—' or a == '—' or pts == '—':
        continue
        
    gp = int(gp)
    g = int(g)
    a = int(a)
    pts = int(pts)
    
    # Skip if no games played
    if gp == 0:
        continue
        
    # Calculate per-game stats
    avg_g = g / gp
    avg_a = a / gp
    avg_pts = pts / gp
    
    # Calculate deviation (in absolute terms)
    dev_g = abs(avg_g - career_avg_g)
    dev_a = abs(avg_a - career_avg_a)
    dev_pts = abs(avg_pts - career_avg_pts)
    
    # Set threshold (e.g., 2x the standard deviation of career stats)
    # Since we don't have full distribution, use a heuristic: 2x the career average as threshold
    threshold_g = 2 * career_avg_g
    threshold_a = 2 * career_avg_a
    threshold_pts = 2 * career_avg_pts
    
    # Check if any metric exceeds threshold
    if dev_g > threshold_g or dev_a > threshold_a or dev_pts > threshold_pts:
        deviating_seasons.append(row['Season'])

# Output the result
print(f"Final Answer: {', '.join(deviating_seasons)}")