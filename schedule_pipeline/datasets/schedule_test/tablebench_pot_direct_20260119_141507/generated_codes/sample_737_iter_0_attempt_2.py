import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average viewership
average_viewership = df['viewers (in millions)'].mean()
print(f"Final Answer: {average_viewership:.2f}")