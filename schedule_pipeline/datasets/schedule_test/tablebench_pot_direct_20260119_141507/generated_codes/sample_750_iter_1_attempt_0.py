import pandas as pd

df = pd.read_csv('table.csv')
# Find the average finish position in 2004
avg_finish_2004 = df[df['year'] == '2004']['avg finish'].values[0]
print(f"Final Answer: {avg_finish_2004}")