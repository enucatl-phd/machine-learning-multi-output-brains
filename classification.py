import pandas as pd
import click
import matplotlib.pyplot as plt
import seaborn as sns


@click.command()
@click.argument("features")
def main(features):
    df = pd.read_csv(features)
    print(df)
    plt.figure()
    g = sns.PairGrid(df, diag_sharey=True)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_upper(plt.scatter)
    g.map_diag(sns.kdeplot, lw=3)
    plt.show()
    plt.ion()
    input()


if __name__ == "__main__":
    main()
