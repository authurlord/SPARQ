import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Drop the row with None rank (Normal / Avg flow)
df = df.dropna(subset=['Rank'])

# Convert columns to numeric
df['Level at Trent Bridge m'] = pd.to_numeric(df['Level at Trent Bridge m'])
df['Peak Flow m3/s'] = pd.to_numeric(df['Peak Flow m3/s'].str.replace(',', ''))

# Scatter plot to show the relationship
plt.scatter(df['Level at Trent Bridge m'], df['Peak Flow m3/s'])
plt.title('Peak Flow vs Water Level at Trent Bridge')
plt.xlabel('Water Level (m)')
plt.ylabel('Peak Flow (m³/s)')
plt.grid(True)
plt.show()

# Calculate the slope of the trend (approximate change in Peak Flow per meter increase in Level)
slope = (df['Peak Flow m3/s'].max() - df['Peak Flow m3/s'].min()) / (df['Level at Trent Bridge m'].max() - df['Level at Trent Bridge m'].min())
print(f"Final Answer: {slope:.1f}")