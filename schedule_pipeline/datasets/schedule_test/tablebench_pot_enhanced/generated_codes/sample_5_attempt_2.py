import pandas as pd

df = pd.read_csv('table.csv')
# Clean the '2001 general' column by extracting numeric values
df['2001 general'] = df['2001 general'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
# Calculate the mean
mean_2001_general = df['2001 general'].mean()
print(f"Final Answer: {mean_2001_general:.1f}")