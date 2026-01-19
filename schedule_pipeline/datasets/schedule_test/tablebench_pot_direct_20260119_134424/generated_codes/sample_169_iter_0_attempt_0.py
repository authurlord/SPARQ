import pandas as pd

df = pd.read_csv('table.csv')
# Convert numeric columns to float
numeric_columns = ['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)']
df[numeric_columns] = df[numeric_columns].astype(float)

# Calculate correlation with 'earnings per share (p)'
correlations = df[numeric_columns].corr()['earnings per share (p)'].abs().sort_values(ascending=False)

# Exclude 'earnings per share (p)' itself
correlations = correlations.drop('earnings per share (p)')

# Get the factor with the highest correlation
main_factor = correlations.idxmax()
print(f"Final Answer: {main_factor}")