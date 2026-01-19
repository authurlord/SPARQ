import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert necessary columns to numeric
df['half-life (s)'] = pd.to_numeric(df['half - life (s)'], errors='coerce')
df['decay constant (s 1 )'] = pd.to_numeric(df['decay constant (s 1 )'], errors='coerce')
df['yield , neutrons per fission'] = pd.to_numeric(df['yield , neutrons per fission'], errors='coerce')
df['fraction'] = pd.to_numeric(df['fraction'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Calculate correlation coefficients
corr_half_life = df['half-life (s)'].corr(df['fraction'])
corr_decay_const = df['decay constant (s 1 )'].corr(df['fraction'])
corr_yield = df['yield , neutrons per fission'].corr(df['fraction'])

# Check for significant influence (absolute correlation > 0.5)
if abs(corr_half_life) > 0.5 or abs(corr_decay_const) > 0.5 or abs(corr_yield) > 0.5:
    # Identify which factor has the highest absolute correlation
    correlations = {
        'half-life (s)': abs(corr_half_life),
        'decay constant (s-1)': abs(corr_decay_const),
        'yield, neutrons per fission': abs(corr_yield)
    }
    strongest_factor = max(correlations, key=correlations.get)
    print(f"Final Answer: {strongest_factor}")
else:
    print("Final Answer: no clear impact")