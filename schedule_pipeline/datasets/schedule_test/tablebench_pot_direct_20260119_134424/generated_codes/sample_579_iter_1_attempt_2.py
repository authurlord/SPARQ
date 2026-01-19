import pandas as pd

df = pd.read_csv('table.csv')

# Clean and parse the 'year' column
def parse_year(year_str):
    if isinstance(year_str, str):
        # Split by comma or space and extract first valid year
        parts = [part.strip() for part in year_str.replace(',', ' ').split()]
        for part in parts:
            try:
                year = int(part)
                return year
            except ValueError:
                continue
    return None

# Apply parsing to the year column
df['year'] = df['year'].apply(parse_year)

# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]

# Convert 'quantity' to numeric, coercing errors to NaN
filtered_df['quantity'] = pd.to_numeric(filtered_df['quantity'], errors='coerce')

# Calculate the average quantity
average_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {average_quantity:.1f}")