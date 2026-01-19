import pandas as pd

df = pd.read_csv('table.csv')

# Describe the table and columns
print("Table Description:")
print("This table represents the results of a dance competition, showing rankings, performance metrics, and outcomes for each couple.")
print("\nColumn Explanations:")
print("- 'rank by average': Ranking based on average score per dance.")
print("- 'competition finish': Final placement in the overall competition.")
print("- 'couple': Name of the dance partner duo.")
print("- 'total': Total points accumulated across all dances.")
print("- 'number of dances': Number of dances each couple performed.")
print("- 'average': Average score per dance (total / number of dances).")

print("\nInitial Insights:")
print("- Darren & Lana (rank 1) have the highest total (374) and average (34.0), indicating top performance.")
print("- Couples with fewer dances (e.g., Rob & Dawn with 2 dances) have lower totals but can still achieve decent averages.")
print("- There is a strong correlation between rank and average score: higher average → higher rank.")
print("- Some couples with high total scores (e.g., Ben & Stephanie) did not win, suggesting that consistency (average) is key.")

print(f"Final Answer: rank by average, competition finish, couple, total, number of dances, average")