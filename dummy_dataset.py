import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
np.random.seed(42)

def make_dummy_dataset(n_features, n_samples):
    # Generate the dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.3
    )
    
    # Randomly assign each sample to one of 10 bags
    n_bags = 10
    bags = np.random.randint(0, n_bags, size=X.shape[0])

    data_df = pd.DataFrame(X)
    data_df.columns = data_df.columns.map(str)
    data_df["y"] = y
    data_df["bag"] = bags
    data_df.to_parquet("dummy_dataset.parquet")
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test, bags_train, bags_test = train_test_split(X, y, bags, test_size=0.2)

    # Initialize an array to hold the class proportions for each bag
    proportions = np.zeros((n_bags, 2))

    # Calculate the class proportions for each bag in the training set
    for i in range(n_bags):
        bag_i = np.where(bags_train == i)[0]
        proportions[i][1] = y_train[bag_i].sum() / len(bag_i)

    # Create a dictionary to store the data and proportions for each bag
    data_dict = {
        str(i): (torch.tensor(X_train[bags_train == i], dtype=torch.float64), torch.tensor(proportions[i], dtype=torch.float64))
        for i in range(n_bags)
    }

    return X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, bags_train, bags_test, proportions, data_dict = make_dummy_dataset(517, 1500)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=5000)
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)
    print(f"accuracy={accuracy_score(y_test, pred)} - f1-score={f1_score(y_test, pred)}")


