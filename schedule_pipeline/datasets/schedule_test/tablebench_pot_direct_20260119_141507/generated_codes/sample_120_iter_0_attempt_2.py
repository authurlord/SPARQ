import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the last row (2014) which has all NaN values
df_clean = df.dropna(subset=['Viewers']).copy()
# Extract years and viewership
years = df_clean['Year'].astype(str)
viewership = df_clean['Viewers'].astype(float)

# Create waterfall chart
plt.figure(figsize=(10, 6))
plt.plot(years, viewership, marker='o', linestyle='-', color='b', label='Viewership')
plt.title('Trend in Viewership of the Program')
plt.xlabel('Year')
plt.ylabel('Viewers (in millions)')
plt.grid(True)
plt.xticks(years)
plt.tight_layout()
plt.show()