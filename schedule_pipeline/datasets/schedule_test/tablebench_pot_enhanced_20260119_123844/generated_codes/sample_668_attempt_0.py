import pandas as pd

df = pd.read_csv('table.csv')
# Convert the required columns to numeric
df['area (km square) territorial waters'] = pd.to_numeric(df['area (km square) territorial waters'])
df['percentage of total area (foreez)'] = pd.to_numeric(df['percentage of total area (foreez)'])

# Calculate the correlation coefficient
correlation = df['area (km square) territorial waters'].corr(df['percentage of total area (foreez)'])
print(f"Final Answer: {correlation:.4f}")