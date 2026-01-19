import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer
df['Year'] = pd.to_numeric(df['Year'])
# Calculate the difference in 'Political Rights' from the previous year
df['diff'] = df['Political Rights'].diff()
# Find the first year where the score decreased by at least 2
first_decrease_year = df[df['diff'] <= -2]['Year'].min()
print(f"Final Answer: {first_decrease_year}")