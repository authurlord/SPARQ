import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Extract the relevant columns
weight_size = df['specimen weight / size']
exposure = df['estimated exposure ( mrem ) / hr']

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(weight_size, exposure, color='blue')
plt.title('Relationship between Specimen Weight/Size and Estimated Exposure (mrem/hr)')
plt.xlabel('Specimen Weight / Size')
plt.ylabel('Estimated Exposure (mrem/hr)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()