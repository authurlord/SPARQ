import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'].astype(str).str.contains(r'200[0-7]')) | (df['year'] == '2000, 2012')]
# Extract only the rows with valid years in 2000-2007
filtered_df = filtered_df[~(df['year'].astype(str).str.contains(r'201[2-9]'))]
# Alternatively, parse year correctly — handle string parsing
years = df['year'].astype(str)
valid_years = years.str.contains(r'200[0-7]', na=False)
filtered_df = df[valid_years]
# Calculate average quantity
avg_quantity = filtered_df['quantity'].mean()
print(f"Final Answer: {avg_quantity:.1f}")