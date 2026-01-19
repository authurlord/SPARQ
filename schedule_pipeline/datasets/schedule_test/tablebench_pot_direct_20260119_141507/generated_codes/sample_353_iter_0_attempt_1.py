import pandas as pd

df = pd.read_csv('table.csv')
# Filter counties where per capita income is between 18000 and 27000 (inclusive)
filtered_counties = df[(df['per capita income'].between(18000, 27000))]
count = filtered_counties.shape[0]
print(f"Final Answer: {count}")