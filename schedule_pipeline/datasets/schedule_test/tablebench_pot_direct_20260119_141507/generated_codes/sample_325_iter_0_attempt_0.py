import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Competition is 'Olympic Games' and Event is '1500 m'
olympic_1500m = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == '1500 m')]

# Extract the position value (convert to integer by removing letters)
def parse_position(pos):
    return int(pos.split('th')[0])

# Apply parsing to get the numeric position
olympic_1500m['position_num'] = olympic_1500m['Position'].apply(parse_position)

# Find the row with the best (lowest) position
best_year = olympic_1500m.loc[olympic_1500m['position_num'].idxmin(), 'Year']
print(f"Final Answer: {best_year}")