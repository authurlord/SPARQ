import pandas as pd

df = pd.read_csv('table.csv')

# Check for correlation between vote percentage and result
bottom_two = df[df['result'] == 'bottom two']
safe_couples = df[df['result'] == 'safe']

# Compare vote percentage
if bottom_two['vote percentage'].values[0] < safe_couples['vote percentage'].min():
    print("Final Answer: vote percentage")
else:
    print("Final Answer: no clear impact")