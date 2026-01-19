import pandas as pd

df = pd.read_csv('table.csv')

# Clean and extract the year: handle ranges like "2003, 2012"
def extract_year(year_str):
    years = [int(y.strip()) for y in year_str.split(',')]
    return min(years)  # Use the earliest year in case of range

df['year'] = df['year'].astype(str).apply(extract_year)

# Filter data for years between 2000 and 2007
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]

# Convert 'quantity' to numeric and calculate average
avg_quantity = filtered_df['quantity'].astype(int).mean()

print(f"Final Answer: {avg_quantity:.1f}")