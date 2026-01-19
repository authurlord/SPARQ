import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract numeric weight from "specimen weight / size"
weights = []
for entry in df['specimen weight / size']:
    # Split by ' / ' and take the first part, then convert to float
    weight_part = entry.split(' / ')[0]
    weights.append(float(weight_part))

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(weights, df['estimated exposure ( mrem ) / hr'], alpha=0.7)
plt.title('Relationship between Specimen Weight/Size and Estimated Exposure (mrem/hr)')
plt.xlabel('Specimen Weight (g)')
plt.ylabel('Estimated Exposure (mrem/hr)')
plt.grid(True)
plt.show()