import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Group by county and sum land area
county_land = df.groupby('county')['land ( sqmi )'].sum().sort_values(ascending=False).head(5)
# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(county_land.index, county_land.values, color='skyblue')
plt.title('Top 5 Counties with Highest Total Land Area (in square miles)')
plt.xlabel('County')
plt.ylabel('Total Land Area (sqmi)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()