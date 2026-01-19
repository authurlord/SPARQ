import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['half-life (s)'] = pd.to_numeric(df['half - life (s)'])
df['decay constant (s 1 )'] = pd.to_numeric(df['decay constant (s 1 )'])
df['yield , neutrons per fission'] = pd.to_numeric(df['yield , neutrons per fission'])
df['fraction'] = pd.to_numeric(df['fraction'])

# Calculate correlation with fraction
correlations = {
    'half-life (s)': df['half-life (s)'].corr(df['fraction']),
    'decay constant (s-1)': df['decay constant (s 1 )'].corr(df['fraction']),
    'yield, neutrons per fission': df['yield , neutrons per fission'].corr(df['fraction'])
}

# Check if any correlation is strong (|r| > 0.7)
strong_influence = [key for key, value in correlations.items() if abs(value) > 0.7]

if strong_influence:
    print(f"Final Answer: {', '.join(strong_influence)}")
else:
    print("Final Answer: no clear impact")