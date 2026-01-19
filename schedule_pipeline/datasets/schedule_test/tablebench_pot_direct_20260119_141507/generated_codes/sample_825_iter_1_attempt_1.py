import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract 'p max ( bar )' column and convert to numeric, handling errors
p_max = pd.to_numeric(df['p max ( bar )'], errors='coerce')

# Remove any NaN values due to parsing issues
p_max = p_max.dropna()

# Calculate mean and standard deviation
mean_pmax = p_max.mean()
std_pmax = p_max.std()

print(f"Final Answer: {mean_pmax:.2f}, {std_pmax:.2f}")