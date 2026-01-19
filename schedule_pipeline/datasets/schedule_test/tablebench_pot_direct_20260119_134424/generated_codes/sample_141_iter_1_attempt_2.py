import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract numeric weight from 'specimen weight / size' column
df['weight_numeric'] = df['specimen weight / size'].str.extract(r'(\d+\.?\d*)').astype(float)

# Use 'weight_numeric' and 'estimated exposure ( mrem ) / hr' for plotting
plt.figure(figsize=(10, 6))
plt.scatter(df['weight_numeric'], df['estimated exposure ( mrem ) / hr'], color='blue')
plt.title('Scatter Plot: Specimen Weight vs Estimated Exposure (mrem/hr)')
plt.xlabel('Specimen Weight (g)')
plt.ylabel('Estimated Exposure (mrem/hr)')
plt.grid(True)
plt.show()