import pandas as pd

df = pd.read_csv('table.csv')
# Get total medal counts for Australia and Russia
australia_total = df[df['nation'] == 'australia']['total'].values[0]
russia_total = df[df['nation'] == 'russia']['total'].values[0]

# Compare and answer
if australia_total > russia_total:
    final_answer = "Australia"
else:
    final_answer = "Russia"
print(f"Final Answer: {final_answer}")