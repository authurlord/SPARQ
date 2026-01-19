import pandas as pd

df = pd.read_csv('table.csv')
# Select the top 5 rows (rank 1 to 5) and get the 'screens' column
top_5_screens = df.head(5)['screens']
# Calculate standard deviation
std_screens = top_5_screens.std()
print(f"Final Answer: {std_screens:.1f}")