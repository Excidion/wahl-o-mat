import polars as pl
from plotly import express as px
import yaml
import streamlit as st
from dash_bio import Clustergram
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import umap


def main():
    title = read_config_value("text", "title")
    st.set_page_config(page_title=title)
    st.title(title)
    st.write(read_config_value("text", "intro"))
    election = st.selectbox("Wahl", get_dataset_names())
    df = load_dataset(election)
    st.write(read_config_value("text", "heatmap0"))
    st.plotly_chart(
        plot_heatmap(df), use_container_width=True, config={"displayModeBar": False}
    )
    st.write(read_config_value("text", "heatmap1"))
    st.subheader("Parteienlandkarte")
    st.write(read_config_value("text", "clusters"))
    plot_party_clusters(df)


def plot_heatmap(df):
    matrix = df.pivot(
        index="topic",
        columns="party",
        values="opinion",
        aggregate_function=None,  # raise error on duplicates
    )
    matrix = matrix.to_pandas()
    y = matrix.pop("topic")
    # remove all neutral parties and topics
    matrix = matrix.loc[:, ~(matrix == 0).all(axis=0)]
    matrix = matrix.loc[~(matrix == 0).all(axis=1)]
    # plot
    fig = Clustergram(
        data=matrix,
        column_labels=list(matrix.columns),
        row_labels=list(y),
        # make dedograms disappear
        display_ratio=0,
        line_width=0,
        # color the heatmap
        color_map=[
            [0, "orange"],  # "rgb(217, 95, 2)"],
            [0.5, "black"],  # "rgb(102, 102, 102)"],
            [1, "blue"],  # "rgb(27, 158, 119)"],
        ],
        center_values=False,  # show real data
        # use manhattan distance (bc of binary votes)
        row_dist="cosine",
        col_dist="cosine",
        link_method="ward",
    )
    # remove colorbar
    fig.data[-1].update(showscale=False)
    return fig


def plot_party_clusters(df):
    config = {
        "selectZoom": False,
        "scrollZoom": True,
        "displayModeBar": False,
    }
    dims = st.radio(
        "Dimension der Karte", [2, 3], horizontal=True, format_func=lambda x: f"{x}D"
    )
    st.plotly_chart(
        _plot_party_clusters(df, dims), use_container_width=True, config=config
    )


# @st.cache_data
def _plot_party_clusters(_df, dimensions=2):
    opinions = _df.pivot(
        index="party",
        columns="topic",
        values="opinion",
        aggregate_function=None,  # raise error on duplicates
    )
    opinions = opinions.to_pandas()
    parties = opinions.pop("party")
    pipe = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "umap",
                umap.UMAP(
                    n_components=dimensions,
                    n_neighbors=3,
                    random_state=42,
                    metric="cosine",
                ),
            ),
        ]
    )
    embedding = pipe.fit_transform(X=opinions)
    opinions["x"] = embedding[:, 0]
    opinions["y"] = embedding[:, 1]
    if dimensions == 3:
        opinions["z"] = embedding[:, 2]
    opinions["party"] = parties
    color_discrete_map = {
        "CDU / CSU": "#151518",
        "PIRATEN": "#ff820a",
        "GRÜNE": "#409A3C",
        "AfD": "#009ee0",
        "SPD": "#E3000F",
        "FDP": "#FFED00",
        "DIE LINKE": "#BE3075",
        "Volt": "#562883",
        "Die PARTEI": "#b5152b",
        "BÜNDNIS DEUTSCHLAND": "#a2bbf3",
        "FAMILIE": "#ff6600",
        "FREIE WÄHLER": "#F7A800",
    }
    for party in parties:
        if party not in color_discrete_map:
            color_discrete_map[party] = "#696969"
    plot_args = dict(
        hover_name="party",
        labels={
            "x": "",
            "y": "",
            "z": "",
        },
        color="party",
        color_discrete_map=color_discrete_map,
    )
    if dimensions == 2:
        fig = px.scatter(opinions, x="x", y="y", **plot_args)
    elif dimensions == 3:
        plot_args["labels"]["z"] = ""
        fig = px.scatter_3d(opinions, x="x", y="y", z="z", **plot_args)
    else:
        raise ValueError("Parameter 'dimensions' has to be either 2 or 3.")
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False)
    fig.update_layout(showlegend=False)
    return fig


def get_dataset_names():
    datasets = read_config().get("datasets").keys()
    return [d for d in datasets if d != "_defaults"]


def load_dataset(name):
    settings = load_settings(name)
    df = pl.read_excel(settings.get("file"), **settings.get("kwargs"))
    for k, v in settings.get("columns").items():
        df = df.rename({v: k})
    for col, mapping in settings.get("mapping").items():
        default = mapping.pop("_default", None)
        df = df.with_columns(pl.col(col).replace(mapping, default=default))
    # chop too long party names
    df = df.with_columns(
        pl.when(pl.col("party").str.len_chars() > 20)
        .then(pl.col("party").str.split(by=" ").list.last())
        .otherwise(pl.col("party"))
        .alias("party")
    )
    return df


def load_settings(name):
    config = read_config()
    datasets = config.get("datasets")
    settings = datasets.get("_defaults")
    dataset = datasets.get(name)
    settings.update(dataset)
    return settings


def read_config_value(*args):
    config = read_config()
    for key in args:
        config = config.get(key)
    return config


def read_config():
    with open("config.yml", "r") as infile:
        return yaml.safe_load(infile)


if __name__ == "__main__":
    main()
