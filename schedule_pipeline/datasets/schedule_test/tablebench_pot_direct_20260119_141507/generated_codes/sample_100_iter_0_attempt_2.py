import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for Costa Rica
costa_rica_data = df[df['Country'] == 'Costa Rica'].iloc[0]

# Extract species counts
species = ['Amphibians', 'Birds', 'Mammals', 'Reptile', 'Vascular plants']
values = [costa_rica_data['Amphibians'], costa_rica_data['Birds'], costa_rica_data['Mammals'], 
          costa_rica_data['Reptile'], costa_rica_data['Vascular plants']]

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=species, autopct='%1.1f%%', startangle=90)
plt.title('Proportions of Species in Costa Rica')
plt.show()

print("Final Answer: pie chart generated for Costa Rica species proportions")