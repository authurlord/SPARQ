import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'inhabitants per mep' in descending order and get top 3
top_3 = df.sort_values('inhabitants per mep', ascending=False).head(3)['member state']
print(f"Final Answer: {', '.join(top_3)}")