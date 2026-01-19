import pandas as pd

df = pd.read_csv('table.csv')
# Filter for magnitude >= 7.7
high_magnitude = df[df['Magnitude'] == '7.7']
# Define Asian countries based on the 'Location' column
asian_countries = ['Iran Iran', 'Pakistan Pakistan', 'Philippines Philippines', 'Japan Japan', 'Indonesia Indonesia']
# Filter for locations in Asia
asian_earthquakes = high_magnitude[high_magnitude['Location'].isin(asian_countries)]
# Sum the death toll
total_death_toll = asian_earthquakes['Death toll'].sum()
print(f"Final Answer: {total_death_toll}")