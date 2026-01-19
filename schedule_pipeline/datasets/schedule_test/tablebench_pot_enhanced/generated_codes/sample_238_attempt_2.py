import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter provinces with PCI > 12
filtered_df = df[df['PCI'].astype(float) > 12]
# Convert DC and PSI to float
filtered_df['DC'] = filtered_df['DC'].astype(float)
filtered_df['PSI'] = filtered_df['PSI'].astype(float)

# Plotting PSI vs DC
plt.figure(figsize=(8, 5))
plt.scatter(filtered_df['DC'], filtered_df['PSI'], color='blue', label='Provinces with PCI > 12')
plt.title('PSI vs DC for Provinces with PCI > 12')
plt.xlabel('DC')
plt.ylabel('PSI')
plt.grid(True)
plt.legend()
plt.show()

# Display the filtered data
print("Filtered Data (PCI > 12):")
print(filtered_df[['Province', 'DC', 'PSI']])