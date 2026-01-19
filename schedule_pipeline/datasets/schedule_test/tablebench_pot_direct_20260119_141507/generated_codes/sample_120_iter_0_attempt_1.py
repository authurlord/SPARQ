import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the last row (2014) which has no viewership data
df_filtered = df.dropna(subset=['Viewers']).copy()

# Extract years and viewership
years = df_filtered['Year'].astype(str)
viewership = df_filtered['Viewers'].astype(float)

# Create waterfall chart
plt.figure(figsize=(10, 6))
plt.bar(years, viewership, color='skyblue', edgecolor='black')
plt.title('Trend in Viewership of the Program')
plt.xlabel('Year')
plt.ylabel('Viewers (in millions)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()