import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper sorting and comparison
df['Year'] = pd.to_numeric(df['Year'])

# Calculate the difference in 'Political Rights' from the previous year
df['diff'] = df['Political Rights'].diff()

# Find the first year where the difference is <= -2 (decrease by at least 2)
first_decrease_year = df[df['diff'] <= -2]['Year'].iloc[0]

print(f"Final Answer: {first_decrease_year}")