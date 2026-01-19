import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to float
numerical_columns = ['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)']
df[numerical_columns] = df[numerical_columns].astype(float)

# Compute correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Extract correlation of 'earnings per share (p)' with other columns
target_column = 'earnings per share (p)'
correlations = correlation_matrix[target_column].abs().drop(target_column)

# Find the column with the highest correlation
influential_factor = correlations.idxmax()
print(f"Final Answer: {influential_factor}")