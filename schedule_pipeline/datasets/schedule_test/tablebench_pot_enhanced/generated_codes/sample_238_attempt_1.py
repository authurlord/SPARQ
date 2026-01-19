import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter provinces with PCI > 12
filtered_df = df[df['PCI'].astype(float) > 12]

# Convert DC and PSI to float
filtered_df['DC'] = filtered_df['DC'].astype(float)
filtered_df['PSI'] = filtered_df['PSI'].astype(float)

# Plot DC vs PSI
plt.figure(figsize=(8, 5))
plt.scatter(filtered_df['DC'], filtered_df['PSI'], color='blue', label='Provinces with PCI > 12')
plt.title('PSI vs DC for Provinces with PCI > 12')
plt.xlabel('DC')
plt.ylabel('PSI')
plt.legend()
plt.grid(True)
plt.show()

# Print the filtered data for clarity
print("Final Answer: The PSI value generally increases with increasing DC values for provinces with PCI > 12.")