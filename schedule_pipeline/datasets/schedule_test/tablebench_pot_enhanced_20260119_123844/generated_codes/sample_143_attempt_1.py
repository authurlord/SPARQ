import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'epoch (utc)' to datetime for proper plotting
df['epoch (utc)'] = pd.to_datetime(df['epoch (utc)'], format='%B %d , %Y , %H:%M:%S.%f', errors='coerce')

# Sort by epoch to ensure correct line order
df.sort_values('epoch (utc)', inplace=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['epoch (utc)'], df['periselene (km)'], label='Periselene (km)', marker='o')
plt.plot(df['epoch (utc)'], df['aposelene (km)'], label='Aposelene (km)', marker='s')

plt.title('Periselene and Aposelene Distances Over Time')
plt.xlabel('Epoch (UTC)')
plt.ylabel('Distance (km)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()