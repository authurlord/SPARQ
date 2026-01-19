import pandas as pd

df = pd.read_csv('table.csv')
# Filter for magnitude >= 7.7
high_magnitude = df[df['Magnitude'] == '7.7']
# Filter for Asian locations
asian_locations = ['Iran Iran', 'Pakistan Pakistan', 'Philippines Philippines']
asian_earthquakes = high_magnitude[high_magnitude['Location'].isin(asian_locations)]
# Sum the death toll
total_death_toll = asian_earthquakes['Death toll'].sum()
print(f"Final Answer: {total_death_toll}")