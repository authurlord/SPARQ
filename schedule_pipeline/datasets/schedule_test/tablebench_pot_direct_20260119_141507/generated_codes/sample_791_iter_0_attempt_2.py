import pandas as pd

df = pd.read_csv('table.csv')
# Select top 10 busiest airports (ranks 1 to 10)
top_10 = df.head(10)
# Extract aircraft movements for these airports
aircraft_movements = top_10['aircraft movements'].astype(float)
# Calculate median
median_movements = aircraft_movements.median()
print(f"Final Answer: {median_movements}")