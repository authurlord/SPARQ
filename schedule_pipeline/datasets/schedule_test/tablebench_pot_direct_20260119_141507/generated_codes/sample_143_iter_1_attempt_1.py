import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Parse the epoch column to extract date-time
df['epoch (utc)'] = pd.to_datetime(df['epoch (utc)'], format='%b %d, %Y %H:%M:%S', errors='coerce')

# Convert periselene and aposelene to numeric
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'], errors='coerce')
df['aposelene (km)'] = pd.to_numeric(df['aposelene (km)'], errors='coerce')

# Sort by epoch to ensure chronological order
df = df.sort_values('epoch (utc)')

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(df['epoch (utc)'], df['periselene (km)'], label='Periselene (km)', marker='o')
plt.plot(df['epoch (utc)'], df['aposelene (km)'], label='Aposelene (km)', marker='s')

plt.xlabel('Epoch (UTC)')
plt.ylabel('Distance (km)')
plt.title('Periselene and Aposelene Distances Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()