import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Calvin L. Rampton and Mike Leavitt
rampton = df[df['Governor'] == 'Calvin L. Rampton']
leavitt = df[df['Governor'] == 'Mike Leavitt']

# Compare days served
if int(rampton['Days'].values[0]) > int(leavitt['Days'].values[0]):
    print("Final Answer: Calvin L. Rampton")
else:
    print("Final Answer: Mike Leavitt")