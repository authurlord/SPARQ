import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Category is 'Best Actor' and Year is 2017 or earlier
filtered_df = df[(df['Category'] == 'Best Actor') & (df['Year'].astype(int) <= 2017)]

# Group by 'Nominated work' and count wins (where Result is 'Won')
win_counts = filtered_df[filtered_df['Result'] == 'Won'].groupby('Nominated work').size()

# Find the film with the maximum wins
if not win_counts.empty:
    film_with_most_wins = win_counts.idxmax()
else:
    film_with_most_wins = None

print(f"Final Answer: {film_with_most_wins}")