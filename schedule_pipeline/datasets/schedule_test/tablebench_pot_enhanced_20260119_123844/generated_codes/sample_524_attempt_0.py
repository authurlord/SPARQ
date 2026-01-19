import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced in or before 2008 and not retired before 2008
# We consider aircraft that were in service in 2008 (retired after 2008 or not retired)
in_service_2008 = df[(df['introduced'] <= '2008') & ((df['retired'] == '-') | (df['retired'] > '2008'))]

# Total capacity in 2008
capacity_2008 = in_service_2008['seating'].astype(int).sum()

# Aircraft in service in 2004: only those introduced in or before 2004 and not retired before 2004
in_service_2004 = df[(df['introduced'] <= '2004') & ((df['retired'] == '-') | (df['retired'] > '2004'))]
capacity_2004 = in_service_2004['seating'].astype(int).sum()

# Change in capacity
change = capacity_2008 - capacity_2004

print(f"Final Answer: {change}")