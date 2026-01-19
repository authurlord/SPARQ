import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter provinces with PCI > 12
filtered_df = df[df['PCI'] > 12]

# Extract DC and PSI values
dc_values = filtered_df['DC'].astype(float)
psi_values = filtered_df['PSI'].astype(float)
provinces = filtered_df['Province']

# Plot DC vs PSI
plt.figure(figsize=(8, 5))
plt.scatter(dc_values, psi_values, color='blue', label='Provinces')
for i, province in enumerate(provinces):
    plt.annotate(province, (dc_values.iloc[i], psi_values.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('DC')
plt.ylabel('PSI')
plt.title('PSI vs DC for Provinces with PCI > 12')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Final answer: List of provinces with their DC and PSI values
result = [f"{province} ({dc}, {psi})" for province, dc, psi in zip(provinces, dc_values, psi_values)]
print(f"Final Answer: {', '.join(result)}")