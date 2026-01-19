import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Parse height to extract feet value
df['height_ft'] = df['height ft / m'].str.split('/').str[0].astype(float)

# Sort by height in feet (descending) and get top 5
top_5 = df.sort_values(by='height_ft', ascending=False).head(5)

# Extract floors for top 5 and compute average
average_floors = top_5['floors'].mean()

print(f"Final Answer: {average_floors:.1f}")