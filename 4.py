/// clustering

pip install pandas numpy matplotlib scikit-learn graphviz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import graphviz

# Load the dataset (simulate "readingSkills" dataset)
# Replace this with actual data loading if available
np.random.seed(42)
readingSkills = pd.DataFrame({
    "nativeSpeaker": np.random.choice(["yes", "no"], 200),
    "age": np.random.randint(5, 18, 200),
    "shoeSize": np.random.uniform(20, 45, 200),
    "score": np.random.randint(1, 100, 200),
})

# Select first 105 rows as input data
input_dat = readingSkills.iloc[:105]

# Prepare input features (X) and target variable (y)
X = input_dat[["age", "shoeSize", "score"]]
y = input_dat["nativeSpeaker"]

# Train Decision Tree model
clf = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
clf.fit(X, y)

# Save the decision tree as a PNG file
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, 
                                class_names=clf.classes_, filled=True, rounded=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves as 'decision_tree.png' and 'decision_tree.pdf'

# Optional: Display the tree in Jupyter Notebook
graph.view()



////////////
library(party)
print(head(readingSkills))
input.dat <- readingSkills[c(1:105),]
png(file = "decision_tree.png")
output.tree <- ctree(nativeSpeaker ~ age + shoeSize + score,data = input.dat)
plot(output.tree)
dev.off()

