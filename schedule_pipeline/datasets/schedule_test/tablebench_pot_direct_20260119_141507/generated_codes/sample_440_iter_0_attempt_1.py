import pandas as pd

df = pd.read_csv('table.csv')

# Display basic info and key insights
print("Key Columns:", df.columns.tolist())
print("\nTop companies by sales:")
print(df.nlargest(5, 'sales (billion )')[['company', 'sales (billion )', 'industry']])
print("\nTop companies by profits:")
print(df.nlargest(5, 'profits (billion )')[['company', 'profits (billion )', 'industry']])
print("\nIndustry distribution:")
print(df['industry'].value_counts())
print("\nSummary statistics for key financials:")
print(df[['sales (billion )', 'profits (billion )', 'assets (billion )', 'market value (billion )']].describe())