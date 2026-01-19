import pandas as pd

df = pd.read_csv('table.csv')
# Convert the columns to numeric, handling any potential non-numeric entries
df['area (km square) territorial waters'] = pd.to_numeric(df['area (km square) territorial waters'], errors='coerce')
df['percentage of total area (foreez)'] = pd.to_numeric(df['percentage of total area (foreez)'], errors='coerce')

# Drop rows with missing values in either column
df.dropna(subset=['area (km square) territorial waters', 'percentage of total area (foreez)'], inplace=True)

# Calculate the correlation coefficient
correlation = df['area (km square) territorial waters'].corr(df['percentage of total area (foreez)'])

print(f"Final Answer: {correlation:.4f}")