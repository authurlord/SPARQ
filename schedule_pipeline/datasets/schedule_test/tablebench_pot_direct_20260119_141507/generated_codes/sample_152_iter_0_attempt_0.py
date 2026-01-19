import pandas as pd

df = pd.read_csv('table.csv')

# Check if 'vote percentage' correlates with 'result'
# Group by result and compute average vote percentage
result_vote_avg = df.groupby('result')['vote percentage'].mean()

# Compare the averages
safe_avg = result_vote_avg.get('safe', 0)
bottom_avg = result_vote_avg.get('bottom two', 0)

# If vote percentage differs significantly, it contributes most
if abs(safe_avg - bottom_avg) > 5:
    print("Final Answer: vote percentage")
else:
    print("Final Answer: no clear impact")