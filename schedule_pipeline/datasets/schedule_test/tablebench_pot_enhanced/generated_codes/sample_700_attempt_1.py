import pandas as pd

df = pd.read_csv('table.csv')
# Remove the '%' sign and convert to float for comparison
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)
# Find the year with the highest success rate
max_success_year = df.loc[df['success rate'].idxmax(), 'year']
print(f"Final Answer: {max_success_year}")