import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2009' and '2011' columns to integers
df['2009'] = pd.to_numeric(df['2009'])
df['2011'] = pd.to_numeric(df['2011'])

# Calculate improvement from 2009 to 2011
df['improvement'] = df['2011'] - df['2009']

# Remove the 'world' row as it's not a country
df_countries = df[df['country'] != 'world']

# Sort by improvement in descending order and get top 3
top_3 = df_countries.nlargest(3, 'improvement')['country'].tolist()

print(f"Final Answer: {', '.join(top_3)}")