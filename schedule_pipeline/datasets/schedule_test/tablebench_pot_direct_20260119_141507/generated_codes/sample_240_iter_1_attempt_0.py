import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean 'no of times visited' and 'no of hc climbs' columns
def parse_numeric(x):
    if isinstance(x, str):
        # Handle cases like '2744', '1850 / 1860', '2240'
        if '/' in x:
            return np.mean([int(part) for part in x.split('/')])
        else:
            return int(x)
    return x

df['no of times visited'] = df['no of times visited'].apply(parse_numeric)
df['no of hc climbs'] = df['no of hc climbs'].apply(parse_numeric)

# Drop rows with NaN after parsing
df = df.dropna(subset=['no of times visited', 'no of hc climbs'])

# Calculate correlation
correlation = df['no of times visited'].corr(df['no of hc climbs'])

print(f"Final Answer: No causal influence can be determined; correlation is {correlation:.2f}")