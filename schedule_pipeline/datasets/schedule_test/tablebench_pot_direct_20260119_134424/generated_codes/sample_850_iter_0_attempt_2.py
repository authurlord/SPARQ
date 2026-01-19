import pandas as pd

df = pd.read_csv('table.csv')

# Convert '% of national vote' to float by removing '%' and converting to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the row with the highest percentage of national votes
max_vote_row = df.loc[df['% of national vote'].idxmax()]

# Get the year and the percentage of national vote for the highest year
highest_year = max_vote_row['election']
highest_percentage = max_vote_row['% of national vote']

# Find the previous year's percentage (the year before the highest year)
previous_year_row = df[df['election'] == str(int(highest_year) - 1)]
if not previous_year_row.empty:
    previous_percentage = previous_year_row.iloc[0]['% of national vote']
else:
    previous_percentage = 0  # In case there's no data for the previous year

# Calculate the difference
difference = highest_percentage - previous_percentage

print(f"Final Answer: {highest_year}, {difference:.2f}")