import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to numeric by removing '%' and converting to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Get overall success rate from the 'total' row
overall_success_rate = df[df['year'] == 'total']['success rate'].values[0]

# Define threshold for "unusually high or low" (e.g., more than 10% deviation)
threshold = 10.0

# Find years with success rate deviating more than threshold from overall
outliers = df[df['year'] != 'total']
outliers = outliers[
    (abs(outliers['success rate'] - overall_success_rate) > threshold)
]

# Extract the years
outlier_years = outliers['year'].tolist()
print(f"Final Answer: {', '.join(outlier_years)}")