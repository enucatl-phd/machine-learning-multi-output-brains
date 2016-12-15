from __future__ import division, print_function
import click
import numpy as np
import scipy.stats
import nibabel as nb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss
from sklearn.model_selection import KFold

from healthybrains.inputoutput import id_from_file_name


@click.command()
@click.option(
    "--targets",
    type=click.Path(exists=True),
    default="data/mlp3-targets.csv")
@click.argument(
    "file_names",
    nargs=-1,
    type=click.Path(exists=True))
def main(targets, file_names):
    targets = np.genfromtxt(targets, delimiter=",")
    quantile_threshold = 0.95
    file_names = np.array(file_names)
    file_ids = [id_from_file_name(file_name)
                for file_name in file_names]
    y = np.array([
        targets[file_id - 1, :]
        for file_id in file_ids], dtype=np.bool)
    kf = KFold(3)
    for train_index, test_index in kf.split(file_names):
        train_files = file_names[train_index]
        test_files = file_names[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        fishers = []
        fisher_thresholds = []
        for feature, feature_name in enumerate(["gender", "age", "health"]):
            print(feature_name)
            feature_files_0 = train_files[y_train[:, feature] == 0]
            feature_files_1 = train_files[y_train[:, feature] == 1]
            data_0 = np.stack([
                np.squeeze(nb.load(file_name).get_data())
                for file_name in feature_files_0],
                axis=-1
            )
            data_1 = np.stack([
                np.squeeze(nb.load(file_name).get_data())
                for file_name in feature_files_1],
                axis=-1
            )
            median_0 = np.median(data_0, axis=-1)
            sd_0 = np.std(data_0, dtype=np.float32, axis=-1)
            median_1 = np.median(data_1, axis=-1)
            sd_1 = np.std(data_1, dtype=np.float32, axis=-1)
            fisher = ((median_0 - median_1) ** 2) / (sd_0 + sd_1)
            fisher[~np.isfinite(fisher)] = 0
            fisher_threshold = scipy.stats.mstats.mquantiles(
                fisher[fisher > 0].ravel(),
                prob=quantile_threshold)[0]
            fisher_thresholds.append(fisher_threshold)
            fishers.append(fisher)
        data = []
        for file_name in file_names:
            d = np.squeeze(nb.load(file_name).get_data())
            data.append(d[np.logical_and(np.logical_and(
                fishers[0] > fisher_thresholds[0],
                fishers[1] > fisher_thresholds[1]),
                fishers[2] > fisher_thresholds[2],
            )])
        X = np.vstack(data)
        X_train, X_test = X[train_index], X[test_index]
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        h = 0
        for f in range(3):
            h += hamming_loss(y_test[:, f], y_pred[:, f])
        print(h / 3)


if __name__ == "__main__":
    main()
