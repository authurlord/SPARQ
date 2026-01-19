import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Group by county and sum the land area (in square miles)
county_land = df.groupby('county')['land ( sqmi )'].sum().sort_values(ascending=False).head(5)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(county_land.index, county_land.values, color='skyblue')
plt.title('Top 5 Counties with Highest Total Land Area (in sqmi)')
plt.xlabel('County')
plt.ylabel('Land Area (sqmi)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output the top 5 counties
print(f"Final Answer: {', '.join(county_land.index)}")