import pandas as pd

df = pd.read_csv('table.csv')
# Compute the absolute difference between 2011 and 2008 values
df['diff'] = abs(df['2011 (imf)'] - df['2008 (cia factbook)'])

# Identify countries with large deviations (difference > 10,000)
outliers = df[df['diff'] > 10000]['nation'].tolist()

print(f"Final Answer: {', '.join(outliers)}")