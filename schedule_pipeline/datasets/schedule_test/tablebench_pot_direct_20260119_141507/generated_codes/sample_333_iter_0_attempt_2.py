import pandas as pd
import re

df = pd.read_csv('table.csv')

# Filter rows where Event is 'Javelin throw' and Competition is 'Olympic Games'
olympic_javelin = df[(df['Event'] == 'Javelin throw') & (df['Competition'] == 'Olympic Games')]

# Extract numeric position from 'Position' column (e.g., '21st (q)' -> 21)
def extract_position(pos):
    match = re.search(r'(\d+)(st|nd|rd|th)', pos)
    return int(match.group(1)) if match else None

# Apply extraction and get the row with the lowest position (best rank)
positions = olympic_javelin['Position'].apply(extract_position)
best_rank_row = olympic_javelin.iloc[positions.idxmin()]

# Return the year of that row
final_year = best_rank_row['Year']
print(f"Final Answer: {final_year}")