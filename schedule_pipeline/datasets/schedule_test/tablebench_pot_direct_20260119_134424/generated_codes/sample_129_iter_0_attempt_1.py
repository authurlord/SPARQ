import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Clean and convert 'Earnings (US$)' to numeric
df['Earnings (US$)'] = df['Earnings (US$)'].str.replace(',', '').astype(float)
# Filter data from 2002 to 2010 (exclude Career row)
df_plot = df[df['Year'] != 'Career']
# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(df_plot['Year'], df_plot['Earnings (US$)'], marker='o', linestyle='-', color='b')
plt.title('Earnings (US$) from 2002 to 2010')
plt.xlabel('Year')
plt.ylabel('Earnings (US$)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()