import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'P' column to numeric, treating non-numeric entries as NaN
df['P'] = pd.to_numeric(df['P'], errors='coerce')
# Drop rows where 'P' is NaN and calculate mean
mean_P = df['P'].mean()
print(f"Final Answer: {mean_P:.1f}")