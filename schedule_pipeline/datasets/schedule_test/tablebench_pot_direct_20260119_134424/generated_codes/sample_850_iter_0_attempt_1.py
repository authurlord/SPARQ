import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of national vote' to float by removing '%' and dividing by 100
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the year with the highest % of national vote
max_vote_year = df.loc[df['% of national vote'].idxmax()]
year = max_vote_year['election']
highest_percentage = max_vote_year['% of national vote']

# Find the previous year's percentage
previous_year_row = df[df['election'] == str(int(year) - 1)]
if not previous_year_row.empty:
    previous_percentage = previous_year_row['% of national vote'].values[0]
else:
    previous_percentage = 0  # In case there is no previous year in data

# Calculate the difference
difference = highest_percentage - previous_percentage

print(f"Final Answer: {year}, {difference:.2f}")