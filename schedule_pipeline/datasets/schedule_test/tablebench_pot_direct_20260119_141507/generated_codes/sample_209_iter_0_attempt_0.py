import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert frequency to numeric: '1' -> 1, '1k' -> 1000, '10k' -> 10000, etc.
def parse_frequency(freq):
    if freq.endswith('k'):
        return int(freq[:-1]) * 1000
    elif freq.endswith('m'):
        return int(freq[:-1]) * 1000000
    else:
        return int(freq)

df['frequency (hz)'] = df['frequency (hz)'].apply(parse_frequency)

# Extract the columns for analysis
freq = df['frequency (hz)']
resistance = df['r (Ω / km)']

# Calculate correlation
correlation = np.corrcoef(freq, resistance)[0, 1]

print(f"Final Answer: {correlation:.3f}")