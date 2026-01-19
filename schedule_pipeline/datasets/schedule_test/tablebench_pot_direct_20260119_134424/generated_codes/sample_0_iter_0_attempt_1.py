import pandas as pd

df = pd.read_csv('table.csv')
mean_cyclones = df['tropical cyclones'].mean()
print(f"Final Answer: {mean_cyclones:.1f}")