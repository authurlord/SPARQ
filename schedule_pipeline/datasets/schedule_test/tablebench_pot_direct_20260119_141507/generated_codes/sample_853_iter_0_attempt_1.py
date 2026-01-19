import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'US Chart position' column to extract numeric values
def extract_position(pos):
    match = re.search(r'(\d+)', pos)
    return int(match.group(1)) if match else None

df['cleaned_position'] = df['US Chart position'].apply(extract_position)

# Find the year with the best (lowest) position and worst (highest) position
best_year = df.loc[df['cleaned_position'].idxmin(), 'Year']
worst_year = df.loc[df['cleaned_position'].idxmax(), 'Year']

print(f"Final Answer: {best_year}, {worst_year}")