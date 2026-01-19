import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to float by removing % and converting to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)
# Get overall success rate from total row
overall_success_rate = df.loc[df['year'] == 'total', 'success rate'].iloc[0]
# Define threshold for unusual performance (e.g., more than 10% deviation)
threshold = 10.0
# Find years with unusually high or low success rate
unusual_years = df[df['year'] != 'total']
unusual_years = unusual_years[
    (unusual_years['success rate'] > overall_success_rate + threshold) |
    (unusual_years['success rate'] < overall_success_rate - threshold)
]['year']
print(f"Final Answer: {', '.join(unusual_years)}")