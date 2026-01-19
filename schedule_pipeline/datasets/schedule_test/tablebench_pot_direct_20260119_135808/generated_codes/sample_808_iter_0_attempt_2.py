import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'P' column to numeric, invalid parsing will be set as NaN
df['P'] = pd.to_numeric(df['P'], errors='coerce')
# Calculate mean of non-null values
mean_P = df['P'].mean()
print(f"Final Answer: {mean_P:.1f}")