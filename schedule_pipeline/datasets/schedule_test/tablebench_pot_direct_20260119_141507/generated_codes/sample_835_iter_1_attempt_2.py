import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# The 'against' column has concatenated values like '10221191974...'
# We need to split it into individual values based on space separation
against_values = []
for row in df['against']:
    # Split the string by space and convert each to int
    values = row.split()
    against_values.extend([int(v) for v in values])

# Compute mean and standard deviation
mean_against = np.mean(against_values)
std_against = np.std(against_values)

print(f"Final Answer: {mean_against:.1f}, {std_against:.1f}")