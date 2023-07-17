import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Read the data from a CSV file
store_data = pd.read_csv('/Users/surabhisuman/Downloads/simple_data-1.csv', header=None)

# Replace all nan values with the string "N/A"
store_data = store_data.fillna("N/A")

# Convert the data to a list of lists
transactions = store_data.values.tolist()

# Initialize the TransactionEncoder
encoder = TransactionEncoder()

# Fit the encoder to the transactions and transform the transactions
onehot = encoder.fit_transform(transactions)

# Convert the onehot array to a pandas DataFrame
df = pd.DataFrame(onehot, columns=encoder.columns_)

# Find frequent itemsets
support_threshold = 0.5
frequent_itemsets = apriori(df, min_support=support_threshold, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)

# Filter rules by confidence and lift
confidence_threshold = 1
lift_threshold = 1
filtered_rules = rules[(rules["confidence"] >= confidence_threshold) & (rules["lift"] >= lift_threshold)]

# Create a new DataFrame to store the filtered rules
result_df = pd.DataFrame(columns=["antecedents", "consequents", "confidence", "lift"])

# Extract the antecedents, consequents, confidence, and lift from the filtered rules and store them in result_df
for index, row in filtered_rules.iterrows():
    antecedents = list(row["antecedents"])
    consequents = list(row["consequents"])
    confidence = row["confidence"]
    lift = row["lift"]
    print(lift)
    result_df.loc[index] = [antecedents, consequents, confidence, lift]

print(result_df)
