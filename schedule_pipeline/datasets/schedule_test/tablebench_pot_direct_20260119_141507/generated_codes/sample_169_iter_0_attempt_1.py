import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation analysis
numeric_columns = ['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)']

# Compute correlation matrix
correlation_matrix = df[numeric_columns].corr()

# Get the correlation between 'earnings per share (p)' and other variables
correlations = correlation_matrix['earnings per share (p)'].drop('earnings per share (p)')

# Find the variable with the highest absolute correlation
most_influential = correlations.abs().idxmax()

print(f"Final Answer: {most_influential}")