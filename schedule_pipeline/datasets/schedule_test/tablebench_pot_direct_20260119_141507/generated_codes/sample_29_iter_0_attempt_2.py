import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2012 to 2014 (inclusive)
filtered_df = df[(df['Theme'].str.contains('2012') | df['Theme'].str.contains('2013') | df['Theme'].str.contains('2014'))]
# Extract and sum the 'Films' column (convert to int if possible)
total_films = filtered_df['Films'].astype(str).str.replace(',', '').str.replace('+', '').astype(int).sum()
print(f"Final Answer: {total_films}")