import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (m)' and 'rating' to numeric, handling 'tba' as NaN
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['viewers (m)', 'rating'], inplace=True)

# Calculate correlation coefficient
correlation = df['viewers (m)'].corr(df['rating'])

print(f"Final Answer: {correlation:.2f}")