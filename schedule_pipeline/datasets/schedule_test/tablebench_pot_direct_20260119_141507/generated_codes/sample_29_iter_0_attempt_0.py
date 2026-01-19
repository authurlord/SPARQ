import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2012 to 2014 (based on the first column, which is the year)
filtered_df = df[(df['-'] >= '2012') & (df['-'] <= '2014')]
# Convert 'Films' column to integer and sum
total_films = filtered_df['Films'].astype(int).sum()
print(f"Final Answer: {total_films}")