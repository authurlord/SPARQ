import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row (last row) since it's not a year
df_filtered = df[df['year'] != 'total']

# Convert success rate to float for comparison
df_filtered['success rate'] = df_filtered['success rate'].str.rstrip('%').astype(float)

# Overall success rate
overall_rate = 60.2

# Identify years with unusually high or low success rates
high_years = df_filtered[df_filtered['success rate'] > overall_rate]['year']
low_years = df_filtered[df_filtered['success rate'] < overall_rate]['year']

# Combine results
unusual_years = list(high_years) + list(low_years)
unusual_years = list(set(unusual_years))  # Remove duplicates

print(f"Final Answer: {', '.join(unusual_years)}")