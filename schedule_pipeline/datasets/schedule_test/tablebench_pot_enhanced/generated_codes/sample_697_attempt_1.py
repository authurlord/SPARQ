import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'inhabitants per mep' to numeric for sorting
df['inhabitants per mep'] = pd.to_numeric(df['inhabitants per mep'])
# Sort by 'inhabitants per mep' in descending order
top_3 = df.sort_values(by='inhabitants per mep', ascending=False).head(3)['member state']
print(f"Final Answer: {', '.join(top_3)}")