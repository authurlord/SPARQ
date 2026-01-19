import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Olympic Games and Javelin throw events
olympic_javelin = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == 'Javelin throw')]

# Extract the numeric position (remove letters like 'st' or 'nd')
def extract_position(pos):
    import re
    match = re.search(r'(\d+)', pos)
    return int(match.group(1)) if match else float('inf')

# Apply the function to get the actual rank number
olympic_javelin['position_num'] = olympic_javelin['Position'].apply(extract_position)

# Find the row with the lowest position number (best rank)
best_rank_row = olympic_javelin.loc[olympic_javelin['position_num'].idxmin()]

# Return the year of that event
print(f"Final Answer: {best_rank_row['Year']}")