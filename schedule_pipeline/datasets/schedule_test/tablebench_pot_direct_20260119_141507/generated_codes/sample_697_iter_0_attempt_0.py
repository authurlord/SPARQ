import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'inhabitants per mep' in descending order and get top 3
top_3_states = df.sort_values(by='inhabitants per mep', ascending=False).head(3)['member state'].tolist()
print(f"Final Answer: {', '.join(top_3_states)}")