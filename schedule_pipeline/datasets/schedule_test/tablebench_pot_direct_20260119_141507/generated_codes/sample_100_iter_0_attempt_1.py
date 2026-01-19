import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for Costa Rica
costa_rica_data = df[df['Country'] == 'Costa Rica']

# Extract species counts
species = {
    'Amphibians': int(costa_rica_data['Amphibians'].values[0]),
    'Birds': int(costa_rica_data['Birds'].values[0]),
    'Mammals': int(costa_rica_data['Mammals'].values[0]),
    'Reptile': int(costa_rica_data['Reptile'].values[0])
}

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(species.values(), labels=species.keys(), autopct='%1.1f%%', startangle=90)
plt.title('Proportions of Species in Costa Rica')
plt.show()