import polars as pl
from utils.graph import Grafo

# load dataset and cleanup
initial_df = pl.read_csv("../data/netflix_amazon_disney_titles.csv")
# only care about id, director and cast
df = initial_df.select('show_id', 'director', 'cast')\
                          .drop_nulls()

# transform the strings into lists with the proper formatting
df = df.with_columns(
    pl.col("director")
      .str.split(",")
      .list.eval(
          pl.element().str.strip_chars().str.to_uppercase()
      )
      .alias("director"),
    pl.col("cast")
      .str.split(",")
      .list.eval(
          pl.element().str.strip_chars().str.to_uppercase()
      )
      .alias("cast")
)

# every row is now a director -> cast
df_exploded = df.explode(columns=['director']).explode(columns='cast')


def main():
    print("Hello from netflix-data-graph-analysis!")


if __name__ == "__main__":
    main()
