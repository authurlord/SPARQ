import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter data for years 2002 to 2010
df_filtered = df[df['Year'].str.contains(r'^200[2-9]$', na=False)]

# Convert earnings to numeric, removing commas
df_filtered['Earnings (US$)'] = df_filtered['Earnings (US$)'].str.replace(',', '').astype(float)

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['Year'], df_filtered['Earnings (US$)'], marker='o', linewidth=2, markersize=6)
plt.title('Earnings Received by the Player from 2002 to 2010')
plt.xlabel('Year')
plt.ylabel('Earnings (US$)')
plt.grid(True)
plt.xticks(df_filtered['Year'])
plt.tight_layout()
plt.show()

print(f"Final Answer: Line chart plotted successfully")