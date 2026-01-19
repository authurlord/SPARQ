import pandas as pd

df = pd.read_csv('table.csv')

# Display key components and insights
print("Key Components and Insights:")
print("- Total population: 892 (448 males, 444 females)")
print("- Child population (0-6): 133 (69 males, 64 females)")
print("- Schedule Caste (SC): 713 (355 males, 358 females)")
print("- Literacy rate: 64.30% (males: 67.28%, females: 61.32%)")
print("- Main workers: 0 (no data for either gender)")
print("- Marginal workers: 82 (62 males, 20 females)")
print("Insights: The population is largely SC with balanced gender representation. Males have higher literacy and marginal worker participation. No main workers reported.")