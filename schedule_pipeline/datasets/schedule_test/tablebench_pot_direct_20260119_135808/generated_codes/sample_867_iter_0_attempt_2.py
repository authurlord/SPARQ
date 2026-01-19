import pandas as pd

df = pd.read_csv('table.csv')
# Extract days served for Calvin L. Rampton and Mike Leavitt
calvin_days = df[df['Governor'] == 'Calvin L. Rampton']['Days'].values[0]
mike_days = df[df['Governor'] == 'Mike Leavitt']['Days'].values[0]

# Compare and determine who served longer
if calvin_days > mike_days:
    longer_serving = 'Calvin L. Rampton'
else:
    longer_serving = 'Mike Leavitt'

print(f"Final Answer: {longer_serving}")