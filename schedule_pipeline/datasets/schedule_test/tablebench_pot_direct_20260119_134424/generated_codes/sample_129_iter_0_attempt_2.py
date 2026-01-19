import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Clean the 'Earnings (US$)' column by removing commas and converting to float
df['Earnings (US$)'] = df['Earnings (US$)'].str.replace(',', '').astype(float)
# Filter data for years 2002 to 2010 (exclude 'Career' row)
df_filtered = df[df['Year'] != 'Career']
# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['Year'], df_filtered['Earnings (US$)'], marker='o', linestyle='-', color='b')
plt.title('Earnings (US$) from 2002 to 2010')
plt.xlabel('Year')
plt.ylabel('Earnings (US$)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()