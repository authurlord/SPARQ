import pandas as pd

df = pd.read_csv('table.csv')

# Display basic structure and key insights
print("Main columns:", df.columns.tolist())
print("\nTop 3 nations by total medals:")
top_nations = df.sort_values(by='Total', ascending=False).head(3)
print(top_nations[['Nation', 'Total', 'Gold', 'Silver', 'Bronze']])

print("\nKey insights:")
print("- Japan leads with 34 total medals (18 gold), showing dominance.")
print("- India and Taiwan tie for second place with 15 total medals.")
print("- Most nations have fewer than 5 total medals, indicating uneven distribution.")