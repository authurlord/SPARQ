import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2012 to 2014
filtered_df = df[(df['-'] >= '2012') & (df['-'] <= '2014')]
# Convert 'Films' column to integers, handling comma-separated numbers
filtered_df['Films'] = filtered_df['Films'].str.replace(',', '').astype(int)
# Sum the number of films
total_films = filtered_df['Films'].sum()
print(f"Final Answer: {total_films}")