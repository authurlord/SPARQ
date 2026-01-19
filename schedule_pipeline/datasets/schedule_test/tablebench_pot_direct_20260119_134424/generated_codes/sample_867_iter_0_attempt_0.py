import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Calvin L. Rampton and Mike Leavitt
rampton_days = df[df['Governor'] == 'Calvin L. Rampton']['Days'].values[0]
leavitt_days = df[df['Governor'] == 'Mike Leavitt']['Days'].values[0]

# Compare and determine who served longer
if rampton_days > leavitt_days:
    longer_serving = 'Calvin L. Rampton'
else:
    longer_serving = 'Mike Leavitt'

print(f"Final Answer: {longer_serving}")