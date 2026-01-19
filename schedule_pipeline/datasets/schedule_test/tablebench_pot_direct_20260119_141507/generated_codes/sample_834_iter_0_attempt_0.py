import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'screens' column for the top 5 chains (first 5 rows)
screens_top_5 = df['screens'].head(5)
# Calculate the standard deviation
std_screens = screens_top_5.std()
print(f"Final Answer: {std_screens:.1f}")