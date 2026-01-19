import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer type to avoid string-int comparison errors
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Filter songs by Glen Campbell and years in the 1980s (1980-1989)
glen_campbell_1980s = df[(df['Artist'] == 'Glen Campbell') & (df['Year'].between(1980, 1989))]

# Count the number of such songs
count = len(glen_campbell_1980s)
print(f"Final Answer: {count}")