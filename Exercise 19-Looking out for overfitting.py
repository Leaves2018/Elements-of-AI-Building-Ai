from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

# do not edit this
# create fake data
x, y = make_moons(
    n_samples=500,  # the number of observations
    random_state=42,
    noise=0.3
)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Create a classifier and fit it to our data
knn = KNeighborsClassifier(n_neighbors=133)
knn.fit(x_train, y_train)

# It seems the plotting job is done by ElementsOfAI
# What we need to do in Exercise 19 is only using the knn.score function
# Documents for sklearn.neighbors.KNeighborsClassifier: 
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
print("training accuracy: %f" % knn.score(x_train, y_train))
print("testing accuracy: %f" % knn.score(x_test, y_test))

# What would be a reasonable baseline accuracy your model should outperform in order for it to be considered useful?
# 0.50

# Which of the following values of k do you think was "best"?
# k=42

# Why?
# it gave the highest testing accuracy

# Is it possible to have a higher test set accuracy than training set accuracy?
# yes