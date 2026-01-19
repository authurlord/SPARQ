import pandas as pd

df = pd.read_csv('table.csv')

# Select only numeric columns for correlation
numeric_columns = ['passengers flown', 'employees (average / year)', 'basic eps', 'net profit / loss (sek)']
df_numeric = df[numeric_columns]

# Compute correlation matrix
correlation_matrix = df_numeric.corr()

# Extract correlations between each factor and net profit/loss (excluding itself)
correlations = correlation_matrix['net profit / loss (sek)'].abs().drop('net profit / loss (sek)')

# Identify factors with significant correlation (> 0.3)
significant_factors = []
for factor, corr in correlations.items():
    if corr > 0.3:
        significant_factors.append(factor)

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")