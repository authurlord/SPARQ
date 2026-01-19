import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Drop the row with missing rank (normal/avg flow)
df_clean = df.dropna(subset=['Rank'])

# Convert columns to numeric
df_clean['Level at Trent Bridge m'] = pd.to_numeric(df_clean['Level at Trent Bridge m'])
df_clean['Peak Flow m3/s'] = pd.to_numeric(df_clean['Peak Flow m3/s'])

# Plot scatter plot
plt.scatter(df_clean['Level at Trent Bridge m'], df_clean['Peak Flow m3/s'])
plt.title('Peak Flow vs Water Level at Trent Bridge')
plt.xlabel('Water Level (m)')
plt.ylabel('Peak Flow (m³/s)')
plt.grid(True)
plt.show()

# Print final answer based on observed trend
print("Final Answer: Increases")