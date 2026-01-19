import pandas as pd

df = pd.read_csv('table.csv')

# Filter for magnitude >= 7.7 and locations in Asia
# Define Asian countries based on the data
asian_countries = ['Iran', 'Pakistan', 'Philippines', 'Japan']

# Filter rows
filtered_df = df[(df['Magnitude'].str.contains(r'7\.7', regex=True)) & 
                 (df['Location'].str.contains('|'.join(asian_countries), case=False, na=False))]

# Sum death toll
total_death_toll = filtered_df['Death toll'].sum()

print(f"Final Answer: {total_death_toll}")