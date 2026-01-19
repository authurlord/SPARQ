import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv('table.csv')

# Extract numeric weight from 'specimen weight / size'
def extract_weight(size_str):
    # Match the number before 'g' or 'cm'
    match = re.search(r'^(\d+)', size_str)
    return int(match.group(1)) if match else 0

df['weight'] = df['specimen weight / size'].apply(extract_weight)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['weight'], df['estimated exposure ( mrem ) / hr'], alpha=0.7)
plt.title('Relationship between Specimen Weight/Size and Estimated Exposure (mrem/hr)')
plt.xlabel('Specimen Weight (g)')
plt.ylabel('Estimated Exposure (mrem/hr)')
plt.grid(True)
plt.show()