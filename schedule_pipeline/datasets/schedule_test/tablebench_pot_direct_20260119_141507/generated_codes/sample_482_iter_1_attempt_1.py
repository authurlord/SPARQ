import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter out the last row (Liiga totals)
data_rows = df.iloc[:-1]
totals_row = df.iloc[-1]

# Extract the total stats (Liiga totals)
total_gp_rs = totals_row['Regular season_3']  # GP in Regular season (last column before 'Playoffs')
total_g_rs = totals_row['Regular season_1']
total_a_rs = totals_row['Regular season_2']
total_pts_rs = totals_row['Regular season_4']
total_pim_rs = totals_row['Regular season_3']  # Wait — let's recheck column indices

# Correct column mapping based on actual structure:
# Regular season columns: GP, G, A, Pts, PIM
# Playoff columns: GP, G, A, Pts, PIM

# Extract the column indices for relevant stats
rs_cols = ['Regular season', 'Regular season_1', 'Regular season_2', 'Regular season_3', 'Regular season_4']
playoff_cols = ['Playoffs', 'Playoffs_1', 'Playoffs_2', 'Playoffs_3', 'Playoffs_4']

# Map column names to actual values
rs_cols_names = ['GP', 'G', 'A', 'Pts', 'PIM']
playoff_cols_names = ['GP', 'G', 'A', 'Pts', 'PIM']

# Extract the total values from the last row (Liiga totals)
total_stats = {
    'GP': totals_row['Regular season_3'],  # GP in regular season
    'G': totals_row['Regular season_1'],
    'A': totals_row['Regular season_2'],
    'Pts': totals_row['Regular season_4'],
    'PIM': totals_row['Regular season_3']  # This seems inconsistent — actually, the PIM is in 'Regular season_3'? Let's check the data
}

# Actually, from the data:
# Row: ['Liiga totals', 'Liiga totals', 'Liiga totals', '-', '415', '134', '123', '258', '298', '-', '60', '17', '17', '34', '22']
# So:
# Regular season: GP=415, G=134, A=123, Pts=258, PIM=298
# Playoffs: GP=60, G=17, A=17, Pts=34, PIM=22

# Therefore, correct totals:
total_rs = [415, 134, 123, 258, 298]  # GP, G, A, Pts, PIM
total_playoffs = [60, 17, 17, 34, 22]

# Now go through each season row (except last)
deviations = []

for idx, row in data_rows.iterrows():
    season = row['Season']
    team = row['Team']
    
    # Extract regular season stats
    rs_gp = row['Regular season']
    rs_g = row['Regular season_1']
    rs_a = row['Regular season_2']
    rs_pts = row['Regular season_3']
    rs_pim = row['Regular season_4']
    
    # Extract playoff stats
    pf_gp = row['Playoffs']
    pf_g = row['Playoffs_1']
    pf_a = row['Playoffs_2']
    pf_pts = row['Playoffs_3']
    pf_pim = row['Playoffs_4']
    
    # Calculate deviations (percentage relative to totals)
    deviations_rs = {}
    deviations_pf = {}
    
    # Regular season
    if rs_gp > 0:
        dev_rs_gp = abs(rs_gp - total_rs[0]) / total_rs[0] * 100
        dev_rs_g = abs(rs_g - total_rs[1]) / total_rs[1] * 100
        dev_rs_a = abs(rs_a - total_rs[2]) / total_rs[2] * 100
        dev_rs_pts = abs(rs_pts - total_rs[3]) / total_rs[3] * 100
        dev_rs_pim = abs(rs_pim - total_rs[4]) / total_rs[4] * 100
        deviations_rs = [dev_rs_gp, dev_rs_g, dev_rs_a, dev_rs_pts, dev_rs_pim]
    
    # Playoff
    if pf_gp > 0:
        dev_pf_gp = abs(pf_gp - total_playoffs[0]) / total_playoffs[0] * 100
        dev_pf_g = abs(pf_g - total_playoffs[1]) / total_playoffs[1] * 100
        dev_pf_a = abs(pf_a - total_playoffs[2]) / total_playoffs[2] * 100
        dev_pf_pts = abs(pf_pts - total_playoffs[3]) / total_playoffs[3] * 100
        dev_pf_pim = abs(pf_pim - total_playoffs[4]) / total_playoffs[4] * 100
        deviations_pf = [dev_pf_gp, dev_pf_g, dev_pf_a, dev_pf_pts, dev_pf_pim]
    
    # Flag if any deviation > 15%
    if any(d > 15 for d in deviations_rs) or any(d > 15 for d in deviations_pf):
        deviations.append(season)

# Final answer: list of seasons with significant deviations
print(f"Final Answer: {', '.join(deviations)}")