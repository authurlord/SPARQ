import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (in millions)' to float for numerical analysis
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'], errors='coerce')

# Calculate correlation between episodes and viewership
correlation = df['episodes'].corr(df['viewers (in millions)'])

print(f"Final Answer: {correlation:.2f}")