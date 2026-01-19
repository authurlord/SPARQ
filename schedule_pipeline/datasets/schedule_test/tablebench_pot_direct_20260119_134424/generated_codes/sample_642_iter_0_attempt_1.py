import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'starting weight (kg)' to float
df['starting weight (kg)'] = pd.to_numeric(df['starting weight (kg)'])

# Convert 'percentage lost' to float by removing the % sign and converting to decimal
df['percentage lost'] = pd.to_numeric(df['percentage lost'].str.replace('%', '')) / 100

# Calculate correlation coefficient
correlation = df['starting weight (kg)'].corr(df['percentage lost'])

print(f"Final Answer: {correlation:.4f}")