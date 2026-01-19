import pandas as pd

df = pd.read_csv('table.csv')
# Extract numeric values from '2001 general' column, handling cases like '19.0 (1996)'
df['2001 general'] = df['2001 general'].astype(str).str.extract('(\d+\.\d+|\d+)').astype(float)
# Calculate the mean
mean_2001_general = df['2001 general'].mean()
print(f"Final Answer: {mean_2001_general:.1f}")