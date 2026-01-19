import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Calvin L. Rampton and Mike Leavitt
rampton_days = df[df['Governor'] == 'Calvin L. Rampton']['Days'].iloc[0]
leavitt_days = df[df['Governor'] == 'Mike Leavitt']['Days'].iloc[0]

# Compare and determine who served longer
if rampton_days > leavitt_days:
    print("Final Answer: Calvin L. Rampton")
else:
    print("Final Answer: Mike Leavitt")