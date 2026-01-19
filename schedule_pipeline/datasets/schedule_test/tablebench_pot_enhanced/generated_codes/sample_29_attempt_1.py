import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2012 to 2014
filtered_df = df[df['-'].isin(['2012', '2013', '2014'])]
# Convert 'Films' column to numeric, replacing '700+' with 700 for simplicity
filtered_df['Films'] = filtered_df['Films'].replace('700+', 700).astype(int)
# Calculate total films
total_films = filtered_df['Films'].sum()
print(f"Final Answer: {total_films}")