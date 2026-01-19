import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to float
df['revenue (million)'] = pd.to_numeric(df['revenue (million)'], errors='coerce')
df['profit / (loss) before tax (m)'] = pd.to_numeric(df['profit / (loss) before tax (m)'], errors='coerce')
df['net profit (m)'] = pd.to_numeric(df['net profit (m)'], errors='coerce')
df['earnings per share (p)'] = pd.to_numeric(df['earnings per share (p)'], errors='coerce')

# Drop rows with NaN values in relevant columns
df.dropna(subset=['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)'], inplace=True)

# Calculate correlation matrix
correlation_matrix = df[['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)']].corr()

# Get correlation of each column with 'earnings per share (p)'
correlations_with_eps = correlation_matrix['earnings per share (p)'].abs().sort_values(ascending=False)

# Exclude 'earnings per share (p)' itself and get the top influencing factor
influencing_factor = correlations_with_eps.index[1]  # Second highest (excluding self)

print(f"Final Answer: {influencing_factor}")