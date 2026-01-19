import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the average finish position for 2004
avg_finish_2004 = df[df['year'] == '2004']['avg finish'].iloc[0]

# Forecasted average finish position for 2005 (same as 2004 due to similar race count)
print(f"Final Answer: {avg_finish_2004}")