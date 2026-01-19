import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average number of podiums
average_podiums = df['podiums'].mean()
print(f"Final Answer: {average_podiums:.1f}")