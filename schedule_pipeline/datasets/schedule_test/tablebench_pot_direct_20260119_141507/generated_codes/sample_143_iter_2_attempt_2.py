import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Parse the epoch column into datetime
df['epoch (utc)'] = pd.to_datetime(df['epoch (utc)'], format='%b %d, %Y %H:%M:%S.%f', errors='coerce')

# Convert periselene and aposelene to numeric (they are already strings with numbers)
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'], errors='coerce')
df['aposelene (km)'] = pd.to_numeric(df['aposelene (km)'], errors='coerce')

# Drop rows with missing values due to parsing or conversion
df = df.dropna()

# Create the line chart
plt.figure(figsize=(12, 6))
plt.plot(df['epoch (utc)'], df['periselene (km)'], label='Periselene (km)', marker='o')
plt.plot(df['epoch (utc)'], df['aposelene (km)'], label='Aposelene (km)', marker='s')

plt.xlabel('Epoch (UTC)')
plt.ylabel('Distance (km)')
plt.title('Periselene and Aposelene Distances Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()