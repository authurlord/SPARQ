import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Parse the epoch column to datetime
df['epoch (utc)'] = pd.to_datetime(df['epoch (utc)'], format='%b %d, %Y %H:%M:%S.%f', errors='coerce')

# Sort by epoch to ensure chronological order
df = df.sort_values('epoch (utc)')

# Extract relevant columns
x = df['epoch (utc)']
y_periselene = df['periselene (km)'].astype(float)
y_apselene = df['aposelene (km)'].astype(float)

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(x, y_periselene, label='Periselene (km)', marker='o')
plt.plot(x, y_apselene, label='Aposelene (km)', marker='s')
plt.xlabel('Epoch (UTC)')
plt.ylabel('Distance (km)')
plt.title('Periselene and Aposelene Distances Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()