import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns for correlation
columns = ['passengers flown', 'employees (average / year)', 'net profit / loss (sek)']
correlation_matrix = df[columns].corr()

# Extract absolute correlation values with 'net profit / loss (sek)'
correlations = correlation_matrix['net profit / loss (sek)'].abs()

# Identify factors with significant correlation (threshold of 0.3)
significant_factors = [col for col, corr in correlations.items() if corr > 0.3]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")