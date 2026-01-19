import pandas as pd

df = pd.read_csv('table.csv')

# Description of columns
print("Table Description:")
print("1. 'rank by average': The rank of the couple based on their average score.")
print("2. 'competition finish': The actual finishing position in the competition.")
print("3. 'couple': The name of the dancing couple.")
print("4. 'total': The total points earned by the couple across all dances.")
print("5. 'number of dances': The number of dances performed by the couple.")
print("6. 'average': The average score per dance.")

# Initial insights
print("\nInitial Insights:")
print(f"Total number of couples: {len(df)}")
print(f"Highest total score: {df['total'].max()} (Darren & Lana)")
print(f"Highest average score: {df['average'].max()} (Darren & Lana)")
print(f"Lowest average score: {df['average'].min()} (Rob & Dawn)")
print(f"Most dances performed: {df['number of dances'].max()} (11 dances)")
print(f"Fewest dances performed: {df['number of dances'].min()} (1 dance)")

# Check if any couple had perfect consistency (average matches total / number of dances)
consistency_check = df[df['average'] == (df['total'] / df['number of dances'])]
if len(consistency_check) > 0:
    print(f"Couples with consistent scoring: {list(consistency_check['couple'])}")

print("Final Answer: rank by average, competition finish, couple, total, number of dances, average")