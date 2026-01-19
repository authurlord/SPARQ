import pandas as pd

df = pd.read_csv('table.csv')

# Filter for magnitude >= 7.7
high_magnitude = df[df['Magnitude'] == '7.7']

# List of Asian countries from the data
asian_countries = ['Iran Iran', 'Pakistan Pakistan', 'Philippines Philippines', 'Japan Japan']

# Filter for earthquakes in Asia
asian_earthquakes = high_magnitude[high_magnitude['Location'].isin(asian_countries)]

# Sum the death toll
total_death_toll = asian_earthquakes['Death toll'].astype(int).sum()

print(f"Final Answer: {total_death_toll}")