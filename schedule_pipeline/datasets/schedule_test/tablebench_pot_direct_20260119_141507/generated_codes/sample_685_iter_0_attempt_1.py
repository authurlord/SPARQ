import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'total' and 'hydroelectricity' columns by removing any text like '(2010)' or non-numeric parts
def clean_numeric(x):
    if isinstance(x, str):
        # Remove parentheses and non-numeric characters
        x = ''.join([c for c in x if c.isdigit() or c == '.'])
        return float(x) if x else np.nan
    return x

df['total'] = df['total'].apply(clean_numeric)
df['hydroelectricity'] = df['hydroelectricity'].apply(clean_numeric)

# Compute the correlation coefficient
correlation = df['total'].corr(df['hydroelectricity'])

print(f"Final Answer: {correlation:.3f}")