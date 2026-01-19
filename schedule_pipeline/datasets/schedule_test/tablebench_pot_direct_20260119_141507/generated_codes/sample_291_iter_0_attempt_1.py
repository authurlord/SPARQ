import pandas as pd

df = pd.read_csv('table.csv')
# Filter for magnitude >= 7.7 and location in Asia
# Define Asia countries based on the location strings
asia_countries = ['Iran', 'Pakistan', 'Japan', 'Philippines', 'Indonesia']
filtered_df = df[(df['Magnitude'].str.contains(r'7\.7')) & (df['Location'].str.contains('|'.join(asia_countries), case=False))]

# Sum the death toll
total_death_toll = filtered_df['Death toll'].sum()
print(f"Final Answer: {total_death_toll}")