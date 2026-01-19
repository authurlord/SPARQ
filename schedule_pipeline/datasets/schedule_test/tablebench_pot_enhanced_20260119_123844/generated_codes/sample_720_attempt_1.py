import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2009' and '2011' columns to numeric
df['2009'] = pd.to_numeric(df['2009'])
df['2011'] = pd.to_numeric(df['2011'])

# Calculate improvement from 2009 to 2011
df['improvement'] = df['2011'] - df['2009']

# Drop the 'world' row and sort by improvement in descending order
top_countries = df[df['country'] != 'world'].sort_values(by='improvement', ascending=False).head(3)

# Extract country names
top_3_countries = top_countries['country'].tolist()
print(f"Final Answer: {', '.join(top_3_countries)}")