import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row since it's not a year
df = df[df['year'] != 'total']

# Convert success rate to float for comparison
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Overall success rate
overall_rate = 60.2

# Find years with unusually high or low success rates
high_rates = df[df['success rate'] > overall_rate]['year']
low_rates = df[df['success rate'] < overall_rate]['year']

# Combine results
unusual_years = list(high_rates) + list(low_rates)
unusual_years = list(set(unusual_years))  # Remove duplicates

print(f"Final Answer: {', '.join(unusual_years)}")