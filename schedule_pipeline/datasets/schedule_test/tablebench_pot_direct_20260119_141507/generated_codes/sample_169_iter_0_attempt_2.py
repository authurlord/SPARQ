import pandas as pd

df = pd.read_csv('table.csv')

# Select numerical columns for correlation
numeric_cols = ['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)']
correlation_with_earnings = df[numeric_cols].corr()['earnings per share (p)']

# Find the column with the highest absolute correlation
most_influential = correlation_with_earnings.abs().idxmax()

print(f"Final Answer: {most_influential}")