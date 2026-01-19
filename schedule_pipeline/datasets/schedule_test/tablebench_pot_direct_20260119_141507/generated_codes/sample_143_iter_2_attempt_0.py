import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Parse the epoch (utc) column to datetime
df['epoch (utc)'] = pd.to_datetime(df['epoch (utc)'], format='%b %d, %Y %H:%M:%S', errors='coerce')

# Set up the plot
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Plot periselene and aposelene over time
ax.plot(df['epoch (utc)'], df['periselene (km)'], label='Periselene (km)', marker='o')
ax.plot(df['epoch (utc)'], df['aposelene (km)'], label='Aposelene (km)', marker='s')

# Format the plot
ax.set_xlabel('Epoch (UTC)')
ax.set_ylabel('Distance (km)')
ax.set_title('Periselene and Aposelene Distances Over Time')
ax.legend()
ax.grid(True)

# Improve date formatting on x-axis
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()