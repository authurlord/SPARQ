import pandas as pd

df = pd.read_csv('table.csv')
# Extract relevant columns
level_m = df['Level at Trent Bridge m'].dropna()
peak_flow_m3s = df['Peak Flow m3/s'].dropna()

# Calculate the correlation between water level (m) and peak flow (m³/s)
correlation = level_m.corr(peak_flow_m3s, method='pearson')
print(f"Final Answer: {correlation:.3f}")