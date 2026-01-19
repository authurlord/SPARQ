import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Event is '800 m' and Competition is 'World Championships'
filtered_df = df[(df['Event'] == '800 m') & (df['Competition'] == 'World Championships')]

# Convert position to numeric (e.g., '9th' -> 9)
def parse_position(pos):
    return int(pos.split()[0])

# Apply parsing and find the row with the best (lowest) position
best_position_row = filtered_df.sort_values(by='Position', key=lambda x: parse_position(x['Position']), ascending=True).iloc[0]
best_year = best_position_row['Year']

print(f"Final Answer: {best_year}")