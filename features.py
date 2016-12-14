from __future__ import division, print_function
import nibabel as nb
import numpy as np
import zlib
import click
import io
import os
import csv
from tqdm import tqdm

from healthybrains.inputoutput import id_from_file_name

desikan_label_ids = {
    "left_thalamus": 10,
    "right_thalamus": 49,
    "left_caudate": 11,
    "right_caudate": 50,
    "left_putamen": 12,
    "right_putamen": 51,
    "left_pallidum": 13,
    "right_pallidum": 52,
    "left_hippocampus": 17,
    "right_hippocampus": 53,
    "left_amygdala": 18,
    "right_amygdala": 54,
}
atlas_base = "/tenochtitlan/data/fsl/"


@click.command()
@click.option("--output")
@click.option("--targets")
@click.argument("file_names", nargs=-1)
def main(output, targets, file_names):
    wbits = zlib.MAX_WBITS | 16
    targets = np.genfromtxt(targets, delimiter=",")
    print(targets.shape)
    with open(output, "w") as outfile:
        writer = csv.writer(outfile)
        for file_name in tqdm(file_names):
            decompressor = zlib.decompressobj(wbits)
            infile = open(file_name, "rb")
            stringio = io.BytesIO()
            decompressed = decompressor.decompress(infile.read())
            infile.close()
            stringio.write(decompressed)
            stringio.seek(0)
            name = np.load(stringio)[0]
            file_id = id_from_file_name(name)
            cortex = np.load(stringio)
            cortex[cortex > 12] = 0
            base = os.path.splitext(os.path.basename(name))[0]
            atlas_name = os.path.join(
                atlas_base,
                "{0}_cropped".format(base),
                "{0}_cropped_warped_atlas.nii.gz".format(base)
            )
            original_name = "/tenochtitlan/data/cropped/{0}_cropped.nii.gz".format(
                base
            )
            data = np.squeeze(nb.load(original_name).get_data())
            atlas = nb.load(atlas_name).get_data()
            print(data.shape)
            print(atlas.shape)
            for region, region_id in desikan_label_ids.iteritems():
                region_data = data[atlas == region_id]


if __name__ == "__main__":
    main()
