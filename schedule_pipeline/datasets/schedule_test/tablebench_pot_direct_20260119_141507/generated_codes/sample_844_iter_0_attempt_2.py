import pandas as pd

df = pd.read_csv('table.csv')
australia_total = df[df['nation'] == 'australia']['total'].values[0]
russia_total = df[df['nation'] == 'russia']['total'].values[0]

if australia_total > russia_total:
    result = "Australia"
else:
    result = "Russia"

print(f"Final Answer: {result}")