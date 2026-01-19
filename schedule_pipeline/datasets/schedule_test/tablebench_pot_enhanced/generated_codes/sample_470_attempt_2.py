import pandas as pd

df = pd.read_csv('table.csv')
# Convert GDP columns to numeric
df['2011 (imf)'] = pd.to_numeric(df['2011 (imf)'])
df['2008 (cia factbook)'] = pd.to_numeric(df['2008 (cia factbook)'])

# Calculate the absolute difference
df['difference'] = abs(df['2011 (imf)'] - df['2008 (cia factbook)'])

# Define a threshold for significant deviation (e.g., 5000)
threshold = 5000
deviant_countries = df[df['difference'] > threshold]['nation']

print(f"Final Answer: {', '.join(deviant_countries)}")