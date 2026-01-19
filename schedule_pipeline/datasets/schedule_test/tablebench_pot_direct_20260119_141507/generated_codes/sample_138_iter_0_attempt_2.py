import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert land area to numeric and calculate total land area per county
df['land ( sqmi )'] = pd.to_numeric(df['land ( sqmi )'], errors='coerce')
county_land = df.groupby('county')['land ( sqmi )'].sum().sort_values(ascending=False)

# Get top 5 counties
top_5_counties = county_land.head(5)

# Create bar chart
plt.figure(figsize=(10, 6))
top_5_counties.plot(kind='bar', color='skyblue')
plt.title('Top 5 Counties by Total Land Area (sqmi)')
plt.xlabel('County')
plt.ylabel('Total Land Area (sqmi)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Final Answer: {list(top_5_counties.index)}")