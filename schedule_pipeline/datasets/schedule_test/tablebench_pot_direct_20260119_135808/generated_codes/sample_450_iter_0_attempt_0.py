import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to float by removing % and converting to numeric
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Get overall success rate from the 'total' row
overall_success_rate = df.loc[df['year'] == 'total', 'success rate'].iloc[0]

# Define threshold for "unusually high or low" (e.g., >10% deviation)
threshold = 10.0

# Find years with success rate significantly different from overall
outliers = df[df['year'] != 'total']
high_outliers = outliers[outliers['success rate'] > (overall_success_rate + threshold)]
low_outliers = outliers[outliers['success rate'] < (overall_success_rate - threshold)]

# Combine and get the years
outlier_years = set(high_outliers['year']).union(set(low_outliers['year']))

print(f"Final Answer: {', '.join(outlier_years)}")