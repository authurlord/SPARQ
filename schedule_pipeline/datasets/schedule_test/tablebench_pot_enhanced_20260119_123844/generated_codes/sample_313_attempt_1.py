import pandas as pd

df = pd.read_csv('table.csv')

# Filter for films released in 2017 or earlier and category 'Best Actor'
filtered_df = df[(df['Year'].astype(int) <= 2017) & (df['Category'] == 'Best Actor')]

# Count wins (Result = 'Won') for each nominated work
win_counts = filtered_df[filtered_df['Result'] == 'Won']['Nominated work'].value_counts()

# Get the film with the most wins
most_wins_film = win_counts.idxmax()

print(f"Final Answer: {most_wins_film}")