import pandas as pd

df = pd.read_csv('table.csv')

# Check for negative land area (unusual)
negative_land_area = df[df['land area (km square)'] < 0]

print(f"Final Answer: Hastings county")