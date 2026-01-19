import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of national vote' to float by removing '%' and converting to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the year with the highest percentage of national votes
max_vote_year = df.loc[df['% of national vote'].idxmax()]
highest_percentage = max_vote_year['% of national vote']

# Find the previous year's percentage (the year before the max vote year)
current_year = int(max_vote_year['election'])
previous_year = current_year - 1
previous_row = df[df['election'] == str(previous_year)]

if not previous_row.empty:
    previous_percentage = previous_row['% of national vote'].values[0]
else:
    previous_percentage = 0  # In case there's no previous year data

# Calculate the difference
difference = highest_percentage - previous_percentage

print(f"Final Answer: {max_vote_year['election']}, {difference:.2f}")