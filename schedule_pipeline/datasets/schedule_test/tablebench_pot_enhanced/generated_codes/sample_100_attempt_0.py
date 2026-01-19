import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for Costa Rica
costa_rica_data = df[df['Country'] == 'Costa Rica']
# Select species columns for the pie chart
species = ['Amphibians', 'Birds', 'Mammals', 'Reptile', 'Vascular plants']
values = [int(costa_rica_data[sp].values[0]) for sp in species]

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=species, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Species in Costa Rica')
plt.show()