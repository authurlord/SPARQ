import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric, treating 'n / a' as NaN
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Calculate standard deviation of the average comprehension scores
std_avg = df['average'].std()
print(f"Final Answer: {std_avg:.2f}")