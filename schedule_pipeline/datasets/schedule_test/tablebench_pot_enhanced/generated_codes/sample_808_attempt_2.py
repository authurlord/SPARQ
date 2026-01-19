import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'P' column to numeric, coercing errors to NaN
df['P'] = pd.to_numeric(df['P'], errors='coerce')
# Calculate mean, excluding NaN values
mean_P = df['P'].mean()
print(f"Final Answer: {mean_P:.1f}")