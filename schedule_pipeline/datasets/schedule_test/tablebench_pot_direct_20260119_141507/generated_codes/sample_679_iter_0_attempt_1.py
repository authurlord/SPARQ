import pandas as pd

df = pd.read_csv('table.csv')
# Extract the two columns for correlation
emissions_per_person = df['carbon dioxide emissions per year (tons per person) (2007)']
emissions_per_km2 = df['avg emission per km² of its land (tons)']

# Calculate the correlation coefficient
correlation_coefficient = emissions_per_person.corr(emissions_per_km2)
print(f"Final Answer: {correlation_coefficient:.2f}")