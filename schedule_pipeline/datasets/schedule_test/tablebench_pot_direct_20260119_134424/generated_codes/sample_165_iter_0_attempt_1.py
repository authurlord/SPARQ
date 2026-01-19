import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric for correlation analysis
df['half-life (s)'] = pd.to_numeric(df['half - life (s)'])
df['decay constant (s 1 )'] = pd.to_numeric(df['decay constant (s 1 )'])
df['yield , neutrons per fission'] = pd.to_numeric(df['yield , neutrons per fission'])
df['fraction'] = pd.to_numeric(df['fraction'])

# Calculate correlation with 'fraction'
correlations = df[['half-life (s)', 'decay constant (s 1 )', 'yield , neutrons per fission']].corrwith(df['fraction'])

# Check if any correlation is strong (absolute value > 0.8)
strong_correlation = correlations[abs(correlations) > 0.8]

if len(strong_correlation) > 0:
    # Return the factor(s) with strong correlation
    result = ', '.join(strong_correlation.index)
else:
    result = 'no clear impact'

print(f"Final Answer: {result}")