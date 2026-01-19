import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select only the relevant columns: 1996 general and 2004 european
col_1996 = '1996 general'
col_2004 = '2004 european'

# Extract the data
values_1996 = df[col_1996].dropna().astype(str)
values_2004 = df[col_2004].dropna().astype(str)

# Clean the values: remove any text in parentheses and convert to float
def clean_value(s):
    if isinstance(s, str):
        # Remove text in parentheses like "(1996)" or "(2001)"
        s = s.split('(')[0].strip()
        s = s.split(')')[0].strip()
        return float(s) if s.replace('.', '').isdigit() else np.nan
    return np.nan

cleaned_1996 = [clean_value(x) for x in values_1996]
cleaned_2004 = [clean_value(x) for x in values_2004]

# Ensure both lists are aligned (same length)
# We'll use only the rows where both values are valid
valid_indices = [i for i in range(len(cleaned_1996)) if not np.isnan(cleaned_1996[i]) and not np.isnan(cleaned_2004[i])]

if len(valid_indices) == 0:
    print("Final Answer: 0.0")
else:
    # Extract valid values
    values_1996_valid = [cleaned_1996[i] for i in valid_indices]
    values_2004_valid = [cleaned_2004[i] for i in valid_indices]
    
    # Calculate percentage change
    changes = [(v2 - v1) / v1 * 100 for v1, v2 in zip(values_1996_valid, values_2004_valid)]
    
    # Average the percentage changes
    avg_change = np.mean(changes)
    print(f"Final Answer: {avg_change:.2f}")