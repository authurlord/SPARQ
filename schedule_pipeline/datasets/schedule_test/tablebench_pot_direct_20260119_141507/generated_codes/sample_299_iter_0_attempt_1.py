import pandas as pd
import re

df = pd.read_csv('table.csv')

# Filter rows where Competition is "World Championships" and Event is "800 m"
filtered_df = df[(df['Competition'] == 'World Championships') & (df['Event'] == '800 m')]

# Extract numeric position from "Position" column (e.g., "23rd" -> 23)
def extract_position(pos):
    match = re.search(r'(\d+)(st|nd|rd|th)', pos)
    if match:
        return int(match.group(1))
    return float('inf')  # Use a large number for invalid or missing positions

# Apply the function to get numeric positions
filtered_df['position_num'] = filtered_df['Position'].apply(extract_position)

# Find the year with the best (lowest) position
best_year = filtered_df.loc[filtered_df['position_num'].idxmin(), 'Year']

print(f"Final Answer: {best_year}")