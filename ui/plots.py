import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import streamlit as st
# SCATTER GRID
def plot_grid(df):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["x"], df["y"], s=1, alpha=0.5)
    ax.set_title("Sensor Distribution on Grid")
    ax.set_xlabel("X Grid")
    ax.set_ylabel("Y Grid")
    ax.grid(True)

    st.pyplot(fig)

def plot_heatmap(df):

    # BUILD HEATMAP

    heatmap = df.groupby(["x", "y"])["traffic_level"].mean().unstack()
    data = heatmap.fillna(0).values


    # FIGURE

    fig, ax = plt.subplots(figsize=(10, 8))

    # heatmap image
    im = ax.imshow(
        data,
        origin="lower",
        cmap="YlGnBu"
    )

    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Traffic Level")

    # titles
    ax.set_title("Traffic Heatmap")
    ax.set_xlabel("X Grid")
    ax.set_ylabel("Y Grid")


    # TEXT VALUES INSIDE CELLS
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 0:
                ax.text(
                    j, i,
                    f"{data[i, j]:.0f}",
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='black'
                )

    # STREAMLIT RENDER
    st.pyplot(fig)


# DENSITY MAP
def plot_density(df):

    density = df.groupby(["x", "y"]).size().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(8,6))

    im = ax.imshow(density, origin="lower", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_title("Sensor Density Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    st.pyplot(fig)


# HIGH TRAFFIC
def plot_high_traffic(df):

    high = df[df["traffic_level"] == 2]
    heat_high = high.groupby(["x", "y"]).size().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(8,6))

    im = ax.imshow(heat_high, origin="lower", cmap="Reds")
    plt.colorbar(im, ax=ax)

    ax.set_title("High Traffic Hotspots")

    st.pyplot(fig)

    # 2) Sensors
    ax.scatter(
        df["x"],
        df["y"],
        s=1,
        alpha=0.3,
        color="black",
        label="Sensors"
    )

    # 3) High Traffic
    high_traffic = df[df["traffic_level"] == 2]

    ax.scatter(
        high_traffic["x"],
        high_traffic["y"],
        s=8,
        color="red",
        label="High Traffic"
    )

    # Final touches
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sensor Density")

    ax.set_title("Combined Traffic Map (Sensors + Hotspots)")
    ax.set_xlabel("X Grid")
    ax.set_ylabel("Y Grid")

    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)



# =========================
# FITNESS CURVE
# =========================
def plot_fitness(history):

    fig, ax = plt.subplots()

    ax.plot(history)
    ax.set_title("Fitness Over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")

    st.pyplot(fig)

def plot_combined(df):

    fig, ax = plt.subplots(figsize=(8,6))

    density = df.groupby(["x", "y"]).size().unstack().fillna(0)

    im = ax.imshow(
        density,
        origin="lower",
        cmap="Blues",
        alpha=0.6
    )

    ax.scatter(
        df["x"],
        df["y"],
        s=1,
        alpha=0.3,
        color="black",
        label="Sensors"
    )

    high_traffic = df[df["traffic_level"] == 2]

    ax.scatter(
        high_traffic["x"],
        high_traffic["y"],
        s=8,
        color="red",
        label="High Traffic"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sensor Density")

    ax.set_title("Combined Traffic Map")
    ax.set_xlabel("X Grid")
    ax.set_ylabel("Y Grid")
    ax.legend()

    st.pyplot(fig)