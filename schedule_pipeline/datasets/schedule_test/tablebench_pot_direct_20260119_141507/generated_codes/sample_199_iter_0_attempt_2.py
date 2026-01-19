import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert "deaths" to numeric: replace text with approximate numbers
death_map = {
    'not known': np.nan,
    '100 +': 100,
    '30 +': 30,
    '200 +': 200,
    'none': 0
}
df['deaths_numeric'] = df['deaths'].map(death_map)

# Drop rows where deaths cannot be converted
df = df.dropna(subset=['deaths_numeric'])

# Calculate correlation between "number of major hurricanes" and "deaths_numeric"
correlation = df['number of major hurricanes'].corr(df['deaths_numeric'])

print(f"Final Answer: {correlation:.2f}")