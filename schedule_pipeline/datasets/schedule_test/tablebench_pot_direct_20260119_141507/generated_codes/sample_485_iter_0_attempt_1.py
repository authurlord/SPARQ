import pandas as pd

df = pd.read_csv('table.csv')

# Filter for top speed anomalies (greater than 150 mph)
anomalous_speed = df[df['Top speed'] > 150]

# Filter for number of units anomalies (less than 5 or greater than 30)
anomalous_number = df[(df['Number'] < 5) | (df['Number'] > 30)]

# Combine unique classes that show deviation in either metric
deviant_classes = set()
for idx, row in anomalous_speed.iterrows():
    deviant_classes.add(row['Class'])
for idx, row in anomalous_number.iterrows():
    deviant_classes.add(row['Class'])

# Convert to list and print
final_deviant = list(deviant_classes)
print(f"Final Answer: {', '.join(final_deviant)}")