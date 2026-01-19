import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation
numeric_columns = ['area km 2', 'area % of eu', 'pop density people / km 2', 'population % of eu']
correlation_matrix = df[numeric_columns].corr()

# Extract correlation between 'population % of eu' and the other three variables
pop_corr = correlation_matrix['population % of eu']

# Identify significant correlations (absolute value > 0.3)
significant_factors = []
for col in ['area km 2', 'area % of eu', 'pop density people / km 2']:
    if abs(pop_corr[col]) > 0.3:
        significant_factors.append(col)

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")