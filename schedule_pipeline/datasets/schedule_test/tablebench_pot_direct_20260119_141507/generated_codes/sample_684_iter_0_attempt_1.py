import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'typhus' and 'smallpox' columns by removing non-numeric characters
def clean_numeric(x):
    if isinstance(x, str):
        # Remove non-numeric characters and convert to float
        return float(''.join(char for char in x if char.isdigit() or char == '.'))
    return x

df['typhus'] = df['typhus'].apply(clean_numeric)
df['smallpox'] = df['smallpox'].apply(clean_numeric)

# Compute the correlation coefficient
correlation = df['typhus'].corr(df['smallpox'])

print(f"Final Answer: {correlation:.3f}")