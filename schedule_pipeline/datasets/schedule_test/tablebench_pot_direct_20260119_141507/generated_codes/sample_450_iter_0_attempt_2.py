import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row since it's not a year
df_filtered = df[df['year'] != 'total']

# Convert success rate to numeric (remove % sign and convert)
df_filtered['success rate'] = df_filtered['success rate'].str.replace('%', '').astype(float)

# Overall success rate
overall_rate = 60.2

# Find years with success rate above or below the overall rate
high_success_years = df_filtered[df_filtered['success rate'] > overall_rate]['year']
low_success_years = df_filtered[df_filtered['success rate'] < overall_rate]['year']

# Combine results
unusual_years = high_success_years.tolist() + low_success_years.tolist()
# Remove duplicates and sort
unusual_years = sorted(list(set(unusual_years)))

print(f"Final Answer: {', '.join(unusual_years)}")