import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to numeric by removing % and converting to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Get overall success rate from the total row
overall_success_rate = df.loc[df['year'] == 'total', 'success rate'].values[0]

# Define threshold for "unusually high or low" (e.g., more than 10 percentage points deviation)
threshold = 10.0

# Find years with unusually high or low success rates
unusual_years = df[df['year'] != 'total']
unusual_years = unusual_years[
    (unusual_years['success rate'] > overall_success_rate + threshold) |
    (unusual_years['success rate'] < overall_success_rate - threshold)
]

# Extract the years
result_years = unusual_years['year'].tolist()
print(f"Final Answer: {', '.join(result_years)}")