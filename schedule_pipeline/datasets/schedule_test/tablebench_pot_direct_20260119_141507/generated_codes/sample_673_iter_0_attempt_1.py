import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Remove the last row which is a summary row
df = df.drop(df.index[-1])

# Clean the 'line length (kilometres)' column: remove comma and convert to float
df['line length (kilometres)'] = df['line length (kilometres)'].str.replace(',', '').astype(float)

# Convert 'annual ridership (1998)' to numeric
df['annual ridership (1998)'] = pd.to_numeric(df['annual ridership (1998)'], errors='coerce')

# Compute the correlation coefficient
correlation = df['line length (kilometres)'].corr(df['annual ridership (1998)'])

print(f"Final Answer: {correlation:.3f}")