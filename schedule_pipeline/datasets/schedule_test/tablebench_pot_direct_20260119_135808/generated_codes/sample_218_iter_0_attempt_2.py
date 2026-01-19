import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'Peak Flow m3/s' by removing commas and converting to float
df['Peak Flow m3/s'] = df['Peak Flow m3/s'].str.replace(',', '').astype(float)
# Extract the relevant columns
level_m = df['Level at Trent Bridge m'].astype(float)
peak_flow_m3s = df['Peak Flow m3/s']
# Calculate correlation coefficient
correlation = level_m.corr(peak_flow_m3s)
print(f"Final Answer: {correlation:.2f}")