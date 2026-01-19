import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of national vote' to float by removing the % sign and converting to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the row with the highest % of national vote
max_vote_row = df.loc[df['% of national vote'].idxmax()]
year_with_max_vote = max_vote_row['election']
max_vote_percentage = max_vote_row['% of national vote']

# Find the previous year's vote percentage
current_year_index = df[df['election'] == year_with_max_vote].index[0]
if current_year_index > 0:
    prev_year_vote = df.iloc[current_year_index - 1]['% of national vote']
else:
    prev_year_vote = 0  # No previous year

# Calculate the difference
difference = max_vote_percentage - prev_year_vote

print(f"Final Answer: {year_with_max_vote}, {difference:.2f}")