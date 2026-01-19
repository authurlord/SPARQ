import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Group by county and sum land area
county_land_area = df.groupby('county')['land ( sqmi )'].sum().sort_values(ascending=False)

# Get top 5 counties
top_5_counties = county_land_area.head(5)

# Create bar chart
plt.figure(figsize=(10, 6))
top_5_counties.plot(kind='bar', color='skyblue')
plt.title('Top 5 Counties by Total Land Area (in square miles)')
plt.xlabel('County')
plt.ylabel('Total Land Area (sqmi)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Final Answer: {list(top_5_counties.index)}")