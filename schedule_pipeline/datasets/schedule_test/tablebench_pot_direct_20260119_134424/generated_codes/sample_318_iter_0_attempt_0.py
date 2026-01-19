import pandas as pd

df = pd.read_csv('table.csv')

# Filter for 400m event at European Championships
filtered_df = df[(df['Event'] == '400 m') & (df['Competition'] == 'European Championships')]

# Find the row with the best (lowest) position
best_position_row = filtered_df.loc[filtered_df['Position'].astype(int).idxmin()]

# Extract the year
best_year = best_position_row['Year']

print(f"Final Answer: {best_year}")