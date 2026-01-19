import pandas as pd

df = pd.read_csv('table.csv')

# Select the columns of interest
factors = ['half-life (s)', 'decay constant (s 1 )', 'yield , neutrons per fission']
target = 'fraction'

# Calculate correlation with the target
correlations = df[factors].corrwith(df[target])

# Check if any correlation is strong (absolute value > 0.5)
strong_correlation = correlations[abs(correlations) > 0.5]

if len(strong_correlation) > 0:
    # Return the factor(s) with strong correlation
    result = ', '.join(strong_correlation.index)
else:
    result = "no clear impact"

print(f"Final Answer: {result}")