import pandas as pd

df = pd.read_csv('table.csv')
# Filter provinces with PCI > 12
filtered_df = df[df['PCI'] > 12]
# Display the DC and PSI values for these provinces
print("Provinces with PCI > 12:")
print(filtered_df[['Province', 'DC', 'PSI']])