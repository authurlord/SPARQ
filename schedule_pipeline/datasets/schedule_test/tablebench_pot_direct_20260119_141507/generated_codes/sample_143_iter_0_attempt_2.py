import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert epoch (utc) to datetime
df['epoch (utc)'] = pd.to_datetime(df['epoch (utc)'], format='%b %d, %Y %H:%M:%S.%f', errors='coerce')

# Set the x-axis to epoch (utc), y-axis to periselene and aposelene
plt.figure(figsize=(10, 6))
plt.plot(df['epoch (utc)'], df['periselene (km)'], label='Periselene (km)', marker='o')
plt.plot(df['epoch (utc)'], df['aposelene (km)'], label='Aposelene (km)', marker='s')

plt.title('Periselene and Aposelene Distances Over Time')
plt.xlabel('Epoch (UTC)')
plt.ylabel('Distance (km)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()