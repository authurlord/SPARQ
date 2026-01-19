import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer for proper comparison
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter rows from 2002 to 2005 inclusive
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2005)]

# Sum the mintage and divide by 5
total_mintage = filtered_df['mintage'].sum()
coins_per_person = total_mintage / 5

print(f"Final Answer: {coins_per_person}")