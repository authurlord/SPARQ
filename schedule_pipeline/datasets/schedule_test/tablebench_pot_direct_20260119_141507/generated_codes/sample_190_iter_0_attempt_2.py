import pandas as pd

df = pd.read_csv('table.csv')

# Select numerical columns for correlation
numeric_columns = ['innings', 'runs scored', 'balls faced', 'average', 'sr']
df_numeric = df[numeric_columns].dropna()

# Compute correlation matrix
correlation_matrix = df_numeric.corr()

# Extract correlations between factors and 'average', 'sr'
correlations_with_average = correlation_matrix['average'].abs().sort_values(ascending=False)
correlations_with_sr = correlation_matrix['sr'].abs().sort_values(ascending=False)

# Combine and find top 2 factors (excluding 'average' and 'sr' themselves)
all_correlations = correlations_with_average.drop('average').combine_first(correlations_with_sr.drop('sr'))
top_2_factors = all_correlations.sort_values(ascending=False).head(2).index.tolist()

print(f"Final Answer: {top_2_factors[0]}, {top_2_factors[1]}")