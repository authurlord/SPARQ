import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for Costa Rica
costa_rica_data = df[df['Country'] == 'Costa Rica']
# Define species and their counts
species = ['Amphibians', 'Birds', 'Mammals', 'Reptile', 'Vascular plants']
counts = [int(costa_rica_data[sp].values[0]) for sp in species]
# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=species, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Species in Costa Rica')
plt.show()