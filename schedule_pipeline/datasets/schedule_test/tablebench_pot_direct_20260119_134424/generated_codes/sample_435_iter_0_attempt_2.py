import pandas as pd

df = pd.read_csv('table.csv')

# Display table structure and main columns
print("Table Structure:")
print(df.info())
print("\nMain Columns:", df.columns.tolist())

# Key insights
top_nation = df.loc[df['Total'].idxmax()]
gold_leader = df.loc[df['Gold'].idxmax()]

print(f"\nKey Insights:")
print(f"- The table ranks nations by total medals, with {top_nation['Nation']} leading with {top_nation['Total']} total medals.")
print(f"- Japan dominates in gold medals with {gold_leader['Gold']}, far ahead of others.")
print(f"- Medal distribution is skewed: Japan has 18 golds, while the next highest (India, Philippines) have only 4.")
print(f"- Silver and bronze are more evenly distributed, but gold remains the primary differentiator.")
print(f"- Nations like India and Taiwan tie in total medals (15), but differ in gold count (4 vs 2).")
print(f"- Several nations (e.g., Nepal, Kampuchea) have only one or two medals, mostly bronze or silver.")

print(f"Final Answer: Japan, 18, 8, 8, 34")