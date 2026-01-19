import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Competition is 'Olympic Games' and Event is '1500 m'
olympic_1500m = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == '1500 m')]

# Extract the position value (convert text like '14th' to number)
def extract_position(pos):
    # Remove any text after the digit (like 'th', 'st', etc.)
    import re
    match = re.search(r'\d+', pos)
    return int(match.group()) if match else None

# Apply extraction and get the row with the best (lowest) position
olympic_1500m['position_num'] = olympic_1500m['Position'].apply(extract_position)
best_position_row = olympic_1500m.loc[olympic_1500m['position_num'].idxmin()]

# Get the year of that row
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")