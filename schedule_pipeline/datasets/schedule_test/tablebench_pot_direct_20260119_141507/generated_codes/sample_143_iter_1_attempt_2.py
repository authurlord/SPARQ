import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Parse the epoch column to datetime
df['epoch (utc)'] = pd.to_datetime(df['epoch (utc)'], format='%b %d, %Y %H:%M:%S.%f', errors='coerce')

# Set up the plot
plt.figure(figsize=(12, 6))
plt.plot(df['epoch (utc)'], df['periselene (km)'], label='Periselene (km)', marker='o')
plt.plot(df['epoch (utc)'], df['aposelene (km)'], label='Aposelene (km)', marker='s')

# Format the x-axis to show dates clearly
plt.xlabel('Epoch (UTC)')
plt.ylabel('Distance (km)')
plt.title('Periselene and Aposelene Distances Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()