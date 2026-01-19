import pandas as pd

df = pd.read_csv('table.csv')

# Filter out total rows (rows where 'Club' is a total or label like 'Liverpool Total')
data = df[~df['Club'].str.contains('Total', na=False)]

# Convert 'Apps' and 'Goals' to numeric
data['Apps'] = pd.to_numeric(data['Apps'], errors='coerce')
data['Goals'] = pd.to_numeric(data['Goals'], errors='coerce')

# Drop any rows with NaN values
data.dropna(subset=['Apps', 'Goals'], inplace=True)

# Calculate correlation between Apps and Goals
correlation = data['Apps'].corr(data['Goals'])

print(f"Final Answer: {correlation:.2f}")