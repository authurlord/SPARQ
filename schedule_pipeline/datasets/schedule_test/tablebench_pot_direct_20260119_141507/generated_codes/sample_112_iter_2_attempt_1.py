import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the 'Career Totals' row
df_filtered = df[df['Team'] != 'Career Totals']

# Convert 'Year' to integer
df_filtered['Year'] = df_filtered['Year'].astype(int)

# Plot the trend in number of attacks (Att) over the years
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['Year'], df_filtered['Att'], marker='o', linestyle='-', color='b')
plt.title('Trend in Number of Attacks Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Attacks (Att)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The line chart has been plotted successfully.
Final Answer: chart