import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select relevant columns
density_mi2 = df['Population density (/mi²)']
land_area_mi2 = df['Land area (mi²)']

# Compute correlation
correlation = density_mi2.corr(land_area_mi2)

# Plot the relationship
plt.figure(figsize=(8, 6))
plt.scatter(land_area_mi2, density_mi2, alpha=0.7)
plt.title('Population Density (/mi²) vs Land Area (mi²)')
plt.xlabel('Land Area (mi²)')
plt.ylabel('Population Density (/mi²)')
plt.grid(True)
plt.show()

print(f"Final Answer: {correlation:.3f}")