import pandas as pd

df = pd.read_csv('table.csv')

# Display key components and insights
print("Key Components and Initial Insights:")
print("- Total number of houses: 187")
print("- Total population: 892 (448 males, 444 females)")
print("- Child population (0-6): 133 (69 male, 64 female)")
print("- Schedule Caste: 713 (355 male, 358 female), no Schedule Tribe")
print("- Literacy rate: 64.30% overall (67.28% male, 61.32% female)")
print("- Total workers: 336 (271 male, 65 female)")
print("- Main workers: 254 (all male), marginal workers: 82 (62 male, 20 female)")
print("Insights:")
print("  - Gender balance in population, but male dominance in literacy and employment.")
print("  - Higher male participation in workforce and main employment.")
print("  - No representation of Schedule Tribe.")
Final Answer: Total number of houses: 187, Total population: 892, Child population: 133, Schedule Caste: 713, Literacy rate: 64.30%, Total workers: 336, Main workers: 254, Marginal workers: 82