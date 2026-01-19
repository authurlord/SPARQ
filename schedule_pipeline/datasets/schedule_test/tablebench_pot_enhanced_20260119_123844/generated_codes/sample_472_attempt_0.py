import pandas as pd

df = pd.read_csv('table.csv')
# Identify the district with the highest population density
max_density_row = df.loc[df['pop density (per km2)'].idxmax()]
district_with_anomaly = max_density_row['district']
print(f"Final Answer: {district_with_anomaly}")