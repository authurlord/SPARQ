import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper filtering
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
# Filter rows from 1975 to 1990 inclusive
filtered_df = df[(df['Year'] >= 1975) & (df['Year'] <= 1990)]
# Extract Political Rights scores
political_rights = filtered_df['Political Rights'].dropna()
# Calculate standard deviation
std_dev = political_rights.std()
print(f"Final Answer: {std_dev:.1f}")