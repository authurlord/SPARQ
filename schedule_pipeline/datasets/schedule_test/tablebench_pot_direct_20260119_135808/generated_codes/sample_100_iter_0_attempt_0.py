import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter the row for Costa Rica
costa_rica_data = df[df['Country'] == 'Costa Rica'].iloc[0]
# Exclude the 'Country' column and use the rest as data for the pie chart
species = costa_rica_data.drop('Country').astype(int)
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(species, labels=species.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Species in Costa Rica')
plt.show()