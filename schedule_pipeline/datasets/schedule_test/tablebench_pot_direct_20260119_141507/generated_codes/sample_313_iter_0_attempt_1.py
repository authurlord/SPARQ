import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Best Actor category and years 2017 or earlier
filtered_df = df[(df['Category'] == 'Best Actor') & (df['Year'].astype(int) <= 2017)]

# Count wins (where Result is "Won")
win_counts = filtered_df[filtered_df['Result'] == 'Won'].groupby('Nominated work').size()

# Get the film with the most wins
if not win_counts.empty:
    film_with_most_wins = win_counts.idxmax()
else:
    film_with_most_wins = None

print(f"Final Answer: {film_with_most_wins}")