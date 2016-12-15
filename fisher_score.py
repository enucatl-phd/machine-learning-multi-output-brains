from __future__ import division, print_function
import nibabel as nb
import numpy as np
import click
import os
import csv
import itertools
from tqdm import tqdm

from healthybrains.inputoutput import id_from_file_name


@click.command()
@click.option("--targets")
@click.argument("file_names", nargs=-1)
def main(targets, file_names):
    targets = np.genfromtxt(targets, delimiter=",")
    shape = nb.load(file_names[0]).shape[:-1]
    file_names = np.array(file_names)
    file_ids = [id_from_file_name(file_name)
                for file_name in file_names]
    genders = np.array([
        targets[file_id - 1, 0]
        for file_id in file_ids], dtype=np.bool)
    for i in range(2):
        gender_files = file_names[genders == i]
        data = np.stack([
            np.squeeze(nb.load(file_name).get_data())
            for file_name in gender_files],
            axis=-1
        )
        median = np.median(data, axis=-1)
        sd = np.std(data, dtype=np.float32, axis=-1)
        np.savez("data/median_{0}.npz".format(i), median)
        np.savez("data/sd_{0}.npz".format(i), sd)

if __name__ == "__main__":
    main()
