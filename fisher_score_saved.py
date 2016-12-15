from __future__ import division
import click
import numpy as np


@click.command()
@click.option("--median1",
              type=click.Path(exists=True),
              default="data/median_1.npz")
@click.option("--median0",
              type=click.Path(exists=True),
              default="data/median_0.npz")
@click.option("--sd0",
              type=click.Path(exists=True),
              default="data/sd_0.npz")
@click.option("--sd1",
              type=click.Path(exists=True),
              default="data/sd_1.npz")
@click.argument("output")
def main(median1, median0, sd1, sd0, output):
    a = np.load(median1)["arr_0"]
    b = np.load(median0)["arr_0"]
    c = np.load(sd0)["arr_0"]
    d = np.load(sd1)["arr_0"]
    fisher = ((a - b) ** 2 / (c + d))
    fisher[~np.isfinite(fisher)] = 0
    np.savez(output, fisher)


if __name__ == "__main__":
    main()
