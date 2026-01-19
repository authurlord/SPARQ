import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the absolute difference between 2011 and 2008 values
df['diff'] = abs(df['2011 (imf)'] - df['2008 (cia factbook)'])
# Identify countries with a significant deviation (difference > 3000)
deviant_countries = df[df['diff'] > 3000]['nation'].tolist()
print(f"Final Answer: {', '.join(deviant_countries)}")