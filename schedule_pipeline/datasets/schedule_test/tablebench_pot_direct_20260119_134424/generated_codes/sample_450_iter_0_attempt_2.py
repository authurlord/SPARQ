import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate from percentage string to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Extract overall success rate from the 'total' row
overall_success_rate = df[df['year'] == 'total']['success rate'].iloc[0]

# Define threshold for "unusually high or low" (e.g., more than 10% deviation)
threshold = 10.0

# Find years with success rate deviating by more than threshold
unusual_years = df[df['year'] != 'total']
unusual_years = unusual_years[
    (unusual_years['success rate'] > overall_success_rate + threshold) |
    (unusual_years['success rate'] < overall_success_rate - threshold)
]

# Extract the years
result_years = unusual_years['year'].tolist()
print(f"Final Answer: {', '.join(result_years)}")