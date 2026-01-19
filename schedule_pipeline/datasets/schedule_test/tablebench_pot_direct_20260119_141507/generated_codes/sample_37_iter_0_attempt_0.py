import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of the 'average' column
mean_average_rating = df['average'].mean()
print(f"Final Answer: {mean_average_rating}")