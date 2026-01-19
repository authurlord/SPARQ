import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where the first column is a ward name (not "ward", "lyrette", etc.)
# We exclude any row where the first value is "ward" or a person's name
valid_rows = df[df['ward'] != 'ward']  # Exclude the header-like row

# Extract the 'bello' column
bello_values = valid_rows['bello']

# Convert to numeric, coercing errors to NaN, then drop invalid entries
bello_numeric = pd.to_numeric(bello_values, errors='coerce')
bello_numeric = bello_numeric.dropna()

# Calculate mean and standard deviation
mean_bello = bello_numeric.mean()
std_bello = bello_numeric.std()

print(f"Final Answer: {mean_bello:.1f}, {std_bello:.1f}")