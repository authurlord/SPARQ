import pandas as pd

df = pd.read_csv('table.csv')
# Drop the 'world' row
df = df[df['country'] != 'world']

# Convert the numeric columns to integers
df[['2009', '2010', '2011']] = df[['2009', '2010', '2011']].apply(pd.to_numeric, errors='coerce')

# Calculate improvement from 2009 to 2011
df['improvement'] = df['2011'] - df['2009']

# Sort by improvement in descending order and get top 3
top_3_countries = df.sort_values(by='improvement', ascending=False).head(3)['country'].tolist()

print(f"Final Answer: {', '.join(top_3_countries)}")