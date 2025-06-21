import polars as pl
from utils.graph import Grafo

def load_data(path):
    # load dataset and cleanup
    initial_df = pl.read_csv(path)
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
    return df


def main():

    df= load_data("../data/netflix_amazon_disney_titles.csv")

    df_exploded = df.explode(columns=['director']).explode(columns='cast')
    # every row is now a director -> cast in df_exploded

    grafo_direcionado = Grafo(direcionado=True)

    #criar o grafo direcionado
    for row in df_exploded.iter_rows(named=True):
        grafo_direcionado.adiciona_aresta(row['cast'], row['director'], peso=1)
    # * ex1: 
    print(f"Quantidade de vértices do grafo direcionado: {grafo_direcionado.ordem}")
    print(f"Quantidade de arestas do grafo direcionado: {grafo_direcionado.tamanho}")

    
    

if __name__ == "__main__":
    main()
