import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['half-life (s)'] = pd.to_numeric(df['half - life (s)'], errors='coerce')
df['decay constant (s 1 )'] = pd.to_numeric(df['decay constant (s 1 )'], errors='coerce')
df['yield , neutrons per fission'] = pd.to_numeric(df['yield , neutrons per fission'], errors='coerce')
df['fraction'] = pd.to_numeric(df['fraction'], errors='coerce')

# Drop rows with NaN values
df.dropna(inplace=True)

# Compute correlation with fraction
correlations = {
    'half-life (s)': df['half-life (s)'].corr(df['fraction']),
    'decay constant (s-1)': df['decay constant (s 1 )'].corr(df['fraction']),
    'yield, neutrons per fission': df['yield , neutrons per fission'].corr(df['fraction'])
}

# Find the factor with the highest absolute correlation
max_corr_factor = max(correlations, key=abs)
max_corr_value = abs(correlations[max_corr_factor])

# Determine significance based on threshold
if max_corr_value > 0.7:
    print(f"Final Answer: {max_corr_factor}")
else:
    print("Final Answer: no clear impact")