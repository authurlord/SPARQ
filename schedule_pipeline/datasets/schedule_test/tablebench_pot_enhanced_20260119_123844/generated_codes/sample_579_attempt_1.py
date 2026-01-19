import pandas as pd

df = pd.read_csv('table.csv')
# Convert year to integer for comparison; some years are listed as ranges like "2003, 2012"
# We'll extract the first year from the string if it's a range
def extract_year(year_str):
    if isinstance(year_str, str):
        return int(year_str.split(',')[0].strip())
    return year_str

df['year'] = df['year'].apply(extract_year)
# Filter data for years between 2000 and 2007
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]
# Calculate average quantity
avg_quantity = filtered_df['quantity'].mean()
print(f"Final Answer: {avg_quantity:.1f}")