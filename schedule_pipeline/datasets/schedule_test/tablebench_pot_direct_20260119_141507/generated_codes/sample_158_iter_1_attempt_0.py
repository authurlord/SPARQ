import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean 'capacity in use' column: replace ',' with '.' and remove '%'
df['capacity in use'] = df['capacity in use'].str.replace(',', '.').str.replace('%', '')

# Convert to float
df['capacity in use'] = pd.to_numeric(df['capacity in use'], errors='coerce')

# Convert 'annual change' to float (already in percentage format, e.g., '9.24%')
df['annual change'] = df['annual change'].str.replace('%', '').astype(float)

# Calculate the correlation between annual change and capacity in use
correlation = df['annual change'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")