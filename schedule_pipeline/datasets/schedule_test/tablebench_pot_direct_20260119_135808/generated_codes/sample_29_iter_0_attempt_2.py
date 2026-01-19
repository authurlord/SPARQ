import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2012 to 2014
filtered_df = df[(df['-'] >= '2012') & (df['-'] <= '2014')]
# Convert 'Films' column to numeric, handling possible non-numeric values like '700+'
filtered_df['Films'] = pd.to_numeric(filtered_df['Films'].str.replace('+', ''), errors='coerce')
total_films = filtered_df['Films'].sum()
print(f"Final Answer: {int(total_films)}")