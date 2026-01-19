import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2012 to 2014
filtered_df = df[(df['Theme'].str.contains('2012') | df['Theme'].str.contains('2013') | df['Theme'].str.contains('2014'))]
# Extract and sum the 'Films' column, converting to numeric (handle non-numeric like '700+')
total_films = filtered_df['Films'].str.replace('+', '').astype(float).sum()
print(f"Final Answer: {int(total_films)}")