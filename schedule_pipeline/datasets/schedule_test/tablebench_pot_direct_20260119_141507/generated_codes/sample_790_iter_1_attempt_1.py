import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric, ignoring errors (replace 'n / a' with NaN)
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Drop rows where 'average' is NaN (i.e., where it was 'n / a')
df_clean = df.dropna(subset=['average'])
# Calculate standard deviation of the cleaned 'average' column
std_avg = df_clean['average'].std()
print(f"Final Answer: {std_avg:.2f}")