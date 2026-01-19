import pandas as pd

df = pd.read_csv('table.csv')
# Calculate median total medals for top 5 nations
top_5_medals = df[df['rank'].astype(int) <= 5]['total'].astype(int)
median_top_5 = top_5_medals.median()

# Calculate median total medals for all nations
all_medals = df['total'].astype(int)
median_all = all_medals.median()

print(f"Final Answer: {median_top_5}, {median_all}")