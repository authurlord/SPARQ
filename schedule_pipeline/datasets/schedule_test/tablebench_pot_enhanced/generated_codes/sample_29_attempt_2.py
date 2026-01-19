import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2012 to 2014
filtered_df = df[df['-'].isin(['2012', '2013', '2014'])]
# Convert 'Films' column to integers, handling '+' in '700+'
total_films = filtered_df['Films'].str.replace('+', '').astype(int).sum()
print(f"Final Answer: {total_films}")