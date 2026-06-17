import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    return df


def create_grid(df, grid_size):


    # 1) Boundary extraction

    min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
    min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
    df = df.dropna(subset=['latitude', 'longitude'])


    # 2) Convert Lat → X
    df["x"] = np.floor(
        (df["latitude"] - min_lat) /
        (max_lat - min_lat + 1e-9) * grid_size
    ).astype(int)

    # 3) Convert Lon → Y
    df["y"] = np.floor(
        (df["longitude"] - min_lon) /
        (max_lon - min_lon + 1e-9) * grid_size
    ).astype(int)

    df["x"] = df["x"].clip(0, grid_size - 1)
    df["y"] = df["y"].clip(0, grid_size - 1)


    # 5) Create heatmap grid
    heatmap = df.groupby(["x", "y"])["traffic_level"].mean().unstack()
    grid = heatmap.fillna(0).values

    return df, grid