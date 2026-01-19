import pandas as pd

df = pd.read_csv('table.csv')
# Filter for magnitude >= 7.7
high_magnitude = df[df['Magnitude'] == '7.7']
# Filter for locations in Asia
asian_locations = high_magnitude[
    (high_magnitude['Location'] == 'Iran Iran') |
    (high_magnitude['Location'] == 'Pakistan Pakistan') |
    (high_magnitude['Location'] == 'Philippines Philippines') |
    (high_magnitude['Location'] == 'Japan Japan') |
    (high_magnitude['Location'] == 'Indonesia Indonesia')
]
# Sum the death toll
total_death_toll = asian_locations['Death toll'].astype(int).sum()
print(f"Final Answer: {total_death_toll}")