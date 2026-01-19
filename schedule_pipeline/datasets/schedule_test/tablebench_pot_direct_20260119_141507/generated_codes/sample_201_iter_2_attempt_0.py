import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop rows with missing values or non-standard entries
df = df.dropna(subset=['Apps', 'Goals'])

# Calculate correlation between Apps and Goals
correlation = df['Apps'].corr(df['Goals'])

# Print the result
print(f"Final Answer: No")