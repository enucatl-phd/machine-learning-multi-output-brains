import click
import numpy as np
import scipy.stats
import nibabel as nb
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.model_selection import KFold

from healthybrains.inputoutput import id_from_file_name


@click.command()
@click.option(
    "--fisher",
    type=click.Path(exists=True),
    default="data/fisher.npz")
@click.option(
    "--targets",
    type=click.Path(exists=True),
    default="data/mlp3-targets.csv")
@click.argument(
    "file_names",
    nargs=-1,
    type=click.Path(exists=True))
def main(fisher, targets, file_names):
    targets = np.genfromtxt(targets, delimiter=",")
    fisher = np.load(fisher)["arr_0"]
    quantile_threshold = 0.95
    fisher_threshold = scipy.stats.mstats.mquantiles(
        fisher[fisher > 0].ravel(),
        prob=quantile_threshold)[0]
    file_ids = [id_from_file_name(file_name)
                for file_name in file_names]
    y = np.array([
        targets[file_id - 1, 0]
        for file_id in file_ids], dtype=np.bool)
    print(fisher_threshold)
    data = []
    for file_name in file_names:
        d = np.squeeze(nb.load(file_name).get_data())
        data.append(d[fisher > fisher_threshold])
    X = np.vstack(data)
    kf = KFold(3)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(hamming_loss(y_test, y_pred))


if __name__ == "__main__":
    main()
