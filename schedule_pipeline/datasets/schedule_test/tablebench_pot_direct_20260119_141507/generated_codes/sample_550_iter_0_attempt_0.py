import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer and sort by year
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.sort_values(by='Year').reset_index(drop=True)

# Calculate the difference in 'Political Rights' from the previous year
df['diff'] = df['Political Rights'].diff()

# Find the first year where the difference is less than or equal to -2
decrease_years = df[df['diff'] <= -2]
if not decrease_years.empty:
    first_decrease_year = decrease_years.iloc[0]['Year']
else:
    first_decrease_year = None

print(f"Final Answer: {first_decrease_year}")