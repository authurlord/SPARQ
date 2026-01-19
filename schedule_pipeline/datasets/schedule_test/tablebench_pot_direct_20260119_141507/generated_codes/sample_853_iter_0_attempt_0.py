import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'US Chart position' column to extract numeric values
def extract_position(pos):
    match = re.search(r'\d+', pos)
    return int(match.group()) if match else None

df['US Chart position'] = df['US Chart position'].apply(extract_position)

# Find the year with the highest and lowest chart positions
max_pos_row = df.loc[df['US Chart position'].idxmax()]
min_pos_row = df.loc[df['US Chart position'].idxmin()]

highest_year = max_pos_row['Year']
lowest_year = min_pos_row['Year']

print(f"Final Answer: {highest_year}, {lowest_year}")