import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['Level at Trent Bridge m'] = pd.to_numeric(df['Level at Trent Bridge m'], errors='coerce')
df['Peak Flow m3/s'] = pd.to_numeric(df['Peak Flow m3/s'], errors='coerce')

# Drop rows with NaN values due to 'Normal / Avg flow' row having missing flow data
df_clean = df.dropna(subset=['Level at Trent Bridge m', 'Peak Flow m3/s'])

# Calculate correlation
correlation = df_clean['Level at Trent Bridge m'].corr(df_clean['Peak Flow m3/s'])

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df_clean['Level at Trent Bridge m'], df_clean['Peak Flow m3/s'], color='blue')
plt.title('Peak Flow (m³/s) vs Water Level (m) at Trent Bridge')
plt.xlabel('Water Level (m)')
plt.ylabel('Peak Flow (m³/s)')
plt.grid(True)
plt.show()

print(f"Final Answer: {correlation:.3f}")