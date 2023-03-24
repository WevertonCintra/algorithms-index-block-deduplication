import recordlinkage
from recordlinkage.datasets import load_febrl3

df, links = load_febrl3(return_links=True)

# Indexation step
indexer = recordlinkage.Index()
indexer.block(left_on="given_name")
candidate_links = indexer.index(df)

# Comparison step
compare = recordlinkage.Compare()

compare.exact("given_name", "given_name", label="given_name")
compare.string("surname", "surname", method="jarowinkler", threshold=0.85, label="surname")
compare.exact("street_number", "street_number", label="street_number")
compare.string("address_1", "address_1", threshold=0.85, label="address_1")
compare.string("address_2", "address_2", threshold=0.85, label="address_2")
compare.string("address_1", "address_2", threshold=0.85, label="address_1_2")
compare.exact("date_of_birth", "date_of_birth", label="date_of_birth")
compare.exact("suburb", "suburb", label="suburb")
compare.exact("postcode", "postcode", label="postcode")
compare.exact("state", "state", label="state")

features = compare.compute(candidate_links, df)

# Classification step
classifier = recordlinkage.ECMClassifier(max_iter=100, binarize=None, atol=10e-4, use_col_names=True)
matches = classifier.fit_predict(features)

confusion_matrix = recordlinkage.confusion_matrix(links, matches)

# Print Metrics
print("Duplicate Pairs:", len(matches))
print("Precision:", recordlinkage.precision(confusion_matrix))
print("Recall:", recordlinkage.recall(confusion_matrix))
print("F-Measure:", recordlinkage.fscore(confusion_matrix))
print("Accuracy", recordlinkage.accuracy(links, matches, candidate_links))
