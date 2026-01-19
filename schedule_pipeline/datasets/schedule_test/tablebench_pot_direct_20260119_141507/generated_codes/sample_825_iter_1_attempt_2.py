import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'p max ( bar )' to numeric, coercing errors to NaN and then dropping invalid entries
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'], errors='coerce')
# Drop any rows with invalid values
df = df.dropna(subset=['p max ( bar )'])
# Calculate mean and standard deviation
mean_p_max = df['p max ( bar )'].mean()
std_p_max = df['p max ( bar )'].std()
print(f"Final Answer: {mean_p_max:.2f}, {std_p_max:.2f}")