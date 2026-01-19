import pandas as pd

df = pd.read_csv('table.csv')
# Extract the two relevant columns
emissions_per_person = df['carbon dioxide emissions per year (tons per person) (2007)']
emission_per_km2 = df['avg emission per km² of its land (tons)']

# Calculate the correlation coefficient
correlation = emissions_per_person.corr(emission_per_km2)
print(f"Final Answer: {correlation:.3f}")