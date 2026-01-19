import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average number of podiums
avg_podiums = df['podiums'].mean()
print(f"Final Answer: {avg_podiums:.1f}")