import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Remove the row with "totals :"
df = df[df['line'] != 'totals :']

# Clean 'line length (kilometres)' by removing spaces and commas and convert to float
df['line length (kilometres)'] = df['line length (kilometres)'].str.replace(',', '').str.replace(' ', '').astype(float)

# Convert 'annual ridership (1998)' to numeric
df['annual ridership (1998)'] = pd.to_numeric(df['annual ridership (1998)'], errors='coerce')

# Compute correlation coefficient
correlation = df['line length (kilometres)'].corr(df['annual ridership (1998)'])

print(f"Final Answer: {correlation:.3f}")