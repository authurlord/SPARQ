import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where Event is '400 m' and Competition is 'European Championships'
filtered_df = df[(df['Event'] == '400 m') & (df['Competition'] == 'European Championships')]

# Convert Position to numeric: remove non-digit characters and convert to int
def parse_position(pos):
    import re
    return int(re.search(r'\d+', pos).group())

# Apply parsing to the Position column
filtered_df['Position_numeric'] = filtered_df['Position'].apply(parse_position)

# Find the row with the minimum (best) position
best_position_row = filtered_df.loc[filtered_df['Position_numeric'].idxmin()]

# Extract the year from that row
best_year = best_position_row['Year']

print(f"Final Answer: {best_year}")