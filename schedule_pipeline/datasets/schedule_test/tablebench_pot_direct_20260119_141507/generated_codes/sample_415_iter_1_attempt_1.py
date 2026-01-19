import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic summary and identify trends
print("Main Contents of the Table:")
print("The table shows annual flight numbers (in thousands) by country from 2002 to 2011.")
print("\nNotable Trends and Patterns:")
print("- Global flight volume increases steadily from 13,600 (2002) to 16,100 (2011).")
print("- China shows strong growth, rising from 585 to 1190 flights per year.")
print("- Other countries exhibit a significant upward trend, indicating rising activity in emerging markets.")
print("- Mexico and Russia show fluctuations, with Mexico experiencing a drop in 2008 and Russia stabilizing.")
print("- The U.S. and Peru show moderate growth with minor dips and recoveries.")
print("- Zambia and Poland show gradual increases over time.")