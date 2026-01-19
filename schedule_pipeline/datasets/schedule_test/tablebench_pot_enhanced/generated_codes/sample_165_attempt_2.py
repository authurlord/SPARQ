import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['half-life (s)'] = pd.to_numeric(df['half - life (s)'])
df['decay constant (s 1 )'] = pd.to_numeric(df['decay constant (s 1 )'])
df['yield , neutrons per fission'] = pd.to_numeric(df['yield , neutrons per fission'])
df['fraction'] = pd.to_numeric(df['fraction'])

# Calculate correlation with fraction
corr_half_life = df['half-life (s)'].corr(df['fraction'])
corr_decay_const = df['decay constant (s 1 )'].corr(df['fraction'])
corr_yield = df['yield , neutrons per fission'].corr(df['fraction'])

# Check for significant influence (absolute correlation > 0.8)
if abs(corr_half_life) > 0.8:
    influence = 'half-life (s)'
elif abs(corr_decay_const) > 0.8:
    influence = 'decay constant (s-1)'
elif abs(corr_yield) > 0.8:
    influence = 'yield, neutrons per fission'
else:
    influence = 'no clear impact'

print(f"Final Answer: {influence}")