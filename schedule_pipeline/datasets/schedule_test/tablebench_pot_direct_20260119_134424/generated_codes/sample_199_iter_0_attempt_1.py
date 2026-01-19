import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'deaths' to numeric, coercing invalid values to NaN
df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce')
# Calculate correlation between 'number of major hurricanes' and 'deaths'
correlation = df['number of major hurricanes'].corr(df['deaths'])
print(f"Final Answer: {correlation:.2f}")