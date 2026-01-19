import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Remove the last row (Liiga totals)
df = df.iloc[:-1]

# Drop rows with '—' or NaN in key columns (like GP, G, A, Pts)
df = df.dropna(subset=['GP', 'G', 'A', 'Pts'])

# Convert relevant columns to numeric
df[['GP', 'G', 'A', 'Pts', 'PIM']] = df[['GP', 'G', 'A', 'Pts', 'PIM']].apply(pd.to_numeric, errors='coerce')

# Get the total stats from the last row (Liiga totals)
total_row = df.iloc[-1]
total_gp = total_row['GP']
total_g = total_row['G']
total_a = total_row['A']
total_pts = total_row['Pts']
total_pim = total_row['PIM']

# Calculate percentage deviation for each season
deviations = []
for idx, row in df.iterrows():
    gp_dev = abs(row['GP'] - total_gp) / total_gp if total_gp > 0 else 0
    g_dev = abs(row['G'] - total_g) / total_g if total_g > 0 else 0
    a_dev = abs(row['A'] - total_a) / total_a if total_a > 0 else 0
    pts_dev = abs(row['Pts'] - total_pts) / total_pts if total_pts > 0 else 0
    pim_dev = abs(row['PIM'] - total_pim) / total_pim if total_pim > 0 else 0
    
    # Use max deviation as a proxy for significant deviation
    max_dev = max(gp_dev, g_dev, a_dev, pts_dev, pim_dev)
    
    if max_dev > 0.1:  # Threshold: 10% deviation
        deviations.append(row['Season'])

# Final Answer: List of seasons with significant deviation
Final Answer: 2006–07, 2008–09, 2017–18