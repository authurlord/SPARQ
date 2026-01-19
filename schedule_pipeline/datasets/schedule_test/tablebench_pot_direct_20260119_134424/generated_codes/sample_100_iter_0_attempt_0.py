import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for Costa Rica
costa_rica_data = df[df['Country'] == 'Costa Rica']
# Select species columns
species_columns = ['Amphibians', 'Birds', 'Mammals', 'Reptile', 'Vascular plants']
species_values = costa_rica_data[species_columns].values.flatten()
# Labels for the pie chart
labels = species_columns
# Draw pie chart
plt.figure(figsize=(8, 8))
plt.pie(species_values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Species in Costa Rica')
plt.show()