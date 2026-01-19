import pandas as pd
import numpy as np

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert frequency column to numeric
def parse_frequency(freq):
    if freq == '1':
        return 1
    elif freq.endswith('k'):
        return int(freq[:-1]) * 1000
    elif freq.endswith('m'):
        return int(freq[:-1]) * 1000000
    else:
        return float(freq)

df['frequency (hz)'] = df['frequency (hz)'].apply(parse_frequency)

# Extract frequency and resistance columns
freq = df['frequency (hz)']
r = df['r (î / km)']

# Calculate Pearson correlation coefficient
correlation = np.corrcoef(freq, r)[0, 1]

print(f"Final Answer: {correlation:.3f}")