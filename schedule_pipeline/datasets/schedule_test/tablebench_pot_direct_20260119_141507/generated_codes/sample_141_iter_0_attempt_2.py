import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv('table.csv')

# Extract numeric weight from "specimen weight / size" (e.g., "1000 g / 8.79 cm" -> 1000)
def extract_weight(size):
    match = re.search(r'(\d+)\s*g', size)
    return int(match.group(1)) if match else None

df['weight_g'] = df['specimen weight / size'].apply(extract_weight)

# Drop rows where weight is missing
df = df.dropna(subset=['weight_g'])

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['weight_g'], df['estimated exposure ( mrem ) / hr'], alpha=0.7)
plt.title('Relationship between Specimen Weight/Size and Estimated Exposure (mrem/hr)')
plt.xlabel('Specimen Weight (g)')
plt.ylabel('Estimated Exposure (mrem/hr)')
plt.grid(True)
plt.show()

print(f"Final Answer: Scatter plot created successfully")