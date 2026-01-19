import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert necessary columns to numeric
df['half-life (s)'] = pd.to_numeric(df['half - life (s)'])
df['decay constant (s 1 )'] = pd.to_numeric(df['decay constant (s 1 )'])
df['yield , neutrons per fission'] = pd.to_numeric(df['yield , neutrons per fission'])
df['fraction'] = pd.to_numeric(df['fraction'])

# Compute correlation with 'fraction'
correlations = {
    'half-life (s)': df['half-life (s)'].corr(df['fraction']),
    'decay constant (s-1)': df['decay constant (s 1 )'].corr(df['fraction']),
    'yield, neutrons per fission': df['yield , neutrons per fission'].corr(df['fraction'])
}

# Check which correlations are significant (absolute value > 0.5)
significant_factors = [factor for factor, corr in correlations.items() if abs(corr) > 0.5]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")