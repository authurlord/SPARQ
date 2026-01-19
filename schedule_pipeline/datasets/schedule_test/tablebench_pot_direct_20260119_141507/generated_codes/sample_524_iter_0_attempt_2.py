import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced before or in 2004 (in service in 2004)
# Exclude retired ones before 2004
# For 2004: aircraft introduced <= 2004 and not retired before 2004
# boeing 737 - 204: retired 1996 → not in 2004
# boeing 737 - 300: retired 2007 → in service until 2007 → in 2004
# airbus a319 - 100: introduced 2004 → in 2004
# airbus a320 - 200: introduced 2008 → not in 2004
# airbus a321 - 200: introduced 2008 → not in 2004

# So in 2004: only a319-100 and a320-200 (but a320-200 not in 2004) → only a319-100 and b737-300
# But b737-300 retired in 2007 → still in 2004 → yes

# 2004 capacity:
capacity_2004 = 0
for _, row in df.iterrows():
    if row['introduced'] <= '2004' and row['retired'] == '-' or row['retired'] > '2004':
        capacity_2004 += int(row['seating'])

# 2008 capacity:
capacity_2008 = 0
for _, row in df.iterrows():
    if row['introduced'] <= '2008' and (row['retired'] == '-' or row['retired'] > '2008'):
        capacity_2008 += int(row['seating'])

# But we must ensure aircraft are active in 2004 and 2008 respectively
# More precise: 
# In 2004: introduced <= 2004 and retired > 2004 or retired == '-'
# In 2008: introduced <= 2008 and retired > 2008 or retired == '-'

# Apply filtering
def is_active_in_year(year):
    return (pd.to_numeric(df['introduced']) <= year) and (df['retired'] == '-' or pd.to_numeric(df['retired']) > year)

# But since years are strings, we do manual string comparison
capacity_2004 = 0
capacity_2008 = 0

for _, row in df.iterrows():
    introduced = row['introduced']
    retired = row['retired']
    seating = row['seating']
    
    # 2004: introduced <= 2004 and retired > 2004 or retired not specified
    if introduced <= '2004' and (retired == '-' or retired > '2004'):
        capacity_2004 += int(seating)
    
    # 2008: introduced <= 2008 and retired > 2008 or retired not specified
    if introduced <= '2008' and (retired == '-' or retired > '2008'):
        capacity_2008 += int(seating)

change = capacity_2008 - capacity_2004
print(f"Final Answer: {change}")