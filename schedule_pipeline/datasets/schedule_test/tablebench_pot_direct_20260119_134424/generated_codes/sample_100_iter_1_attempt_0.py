import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for Costa Rica
costa_rica_data = df[df['Country'] == 'Costa Rica']

# Extract species counts (excluding 'Country' and 'Biodiversity')
species = costa_rica_data.iloc[0][['Amphibians', 'Birds', 'Mammals', 'Reptile', 'Vascular plants']].astype(int)
labels = ['Amphibians', 'Birds', 'Mammals', 'Reptile', 'Vascular Plants']

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(species, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Species in Costa Rica')
plt.show()