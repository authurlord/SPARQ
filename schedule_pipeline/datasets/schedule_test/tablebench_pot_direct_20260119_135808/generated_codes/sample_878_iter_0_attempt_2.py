import pandas as pd

df = pd.read_csv('table.csv')
# Sort by points in descending order
df_sorted = df.sort_values(by='points', ascending=False)
# Get the highest score and the 5th highest score
highest_points = df_sorted.iloc[0]['points']
fifth_highest_points = df_sorted.iloc[4]['points']
# Calculate the difference
difference = int(highest_points) - int(fifth_highest_points)
print(f"Final Answer: {difference}")