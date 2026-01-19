import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'].astype(str).str.startswith('200') & (df['year'].astype(str).str.contains('2000|2001|2003|2007')))]
# Extract and convert quantity to numeric
quantities = pd.to_numeric(df['quantity'], errors='coerce')
# Filter by year range and compute mean
filtered_quantities = quantities[(df['year'].between(2000, 2007))]
mean_quantity = filtered_quantities.mean()
print(f"Final Answer: {mean_quantity:.1f}")