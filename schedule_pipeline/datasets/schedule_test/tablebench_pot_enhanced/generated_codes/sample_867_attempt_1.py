import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Calvin L. Rampton and Mike Leavitt
calvin_days = df[df['Governor'] == 'Calvin L. Rampton']['Days'].values[0]
mike_days = df[df['Governor'] == 'Mike Leavitt']['Days'].values[0]

# Compare and determine who served longer
if calvin_days > mike_days:
    print("Final Answer: Calvin L. Rampton")
else:
    print("Final Answer: Mike Leavitt")