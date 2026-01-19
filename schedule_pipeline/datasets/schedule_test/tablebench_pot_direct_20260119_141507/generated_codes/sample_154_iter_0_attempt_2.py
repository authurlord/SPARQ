import pandas as pd

df = pd.read_csv('table.csv')

# Select the relevant columns
columns_to_check = ['preliminaries', 'interview', 'swimsuit', 'evening gown']
average_col = 'average'

# Calculate correlation with average for each factor
correlations = df[columns_to_check].corrwith(df[average_col])

# Find which factors have correlation absolute value > 0.7
significant_factors = [col for col in columns_to_check if abs(correlations[col]) > 0.7]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")