import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# CORE
from core.data_processing import load_data, create_grid
from core.ga import run_ga
from core.de import run_de
from core.utils import coverage_ratio
# UI
from ui.plots import (
    plot_grid,
    plot_heatmap,
    plot_density,
    plot_high_traffic,
    plot_combined,
    plot_fitness
)
from ui.controls import sidebar_controls

# PAGE CONFIG
st.set_page_config(
    page_title="Sensor Optimization",
    layout="wide"
)

st.title("📡 Sensor Optimization using Evolutionary Algorithms")

# SESSION STATE
if "runs_history" not in st.session_state:
    st.session_state.runs_history = []

if "last_mode" not in st.session_state:
    st.session_state.last_mode = None

# SIDEBAR
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Run Mode",
    ["Single Run", "Compare All (GA vs DE vs HYBRID)"]
)

# RESET ON MODE CHANGE
if st.session_state.last_mode != mode:
    st.session_state.runs_history = []
    st.session_state.last_mode = mode

# Algorithm only in single mode
if mode == "Single Run":
    single_algo = st.sidebar.selectbox(
        "Algorithm",
        ["GA", "DE", "HYBRID"]
    )
else:
    single_algo = None

# GA SETTINGS
st.sidebar.markdown("### 🧬 GA Settings")

selection_method = st.sidebar.selectbox(
    "Selection Method",
    ["tournament", "roulette", "sus", "over"]
)

crossover_type = st.sidebar.selectbox(
    "Crossover Type",
    ["uniform", "onepoint"]
)

mutation_type = st.sidebar.selectbox(
    "Mutation Type",
    ["shift", "reset"]
)

diversity_type = st.sidebar.selectbox(
    "Diversity Method",
    ["basic", "strong"]
)

survivor_method = st.sidebar.selectbox(
    "Survivor Method",
    ["elitism", "rank"]
)

init_method = st.sidebar.selectbox(
    "Initialization",
    ["heuristic", "random"]
)

# =========================
# NEW 🔥 RATES CONTROL
# =========================
crossover_rate = st.sidebar.slider(
    "Crossover Rate",
    0.0, 1.0, 0.7
)

mutation_rate = st.sidebar.slider(
    "Mutation Rate",
    0.0, 1.0, 0.1
)

# GENERAL SETTINGS
pop_size, sensors, generations = sidebar_controls()

radius = st.sidebar.slider("Coverage Radius", 1, 10, 5)
grid_size = st.sidebar.slider("Grid Size", 20, 100, 50)

data_path = st.sidebar.text_input(
    "Dataset Path",
    "data/metr_la_reduced.csv"
)
# LOAD DATA
@st.cache_data
def load_all(path, grid_size):
    df = load_data(path)
    df, grid = create_grid(df, grid_size)
    return df, grid

try:
    df, grid = load_all(data_path, grid_size)
    st.success("✅ Dataset Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# VISUALIZATION
st.subheader("📊 Data Visualization")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Grid",
    "Heatmap",
    "Density",
    "High Traffic",
    "Combined"
])

with tab1:
    plot_grid(df)

with tab2:
    plot_heatmap(df)

with tab3:
    plot_density(df)

with tab4:
    plot_high_traffic(df)

with tab5:
    plot_combined(df)

# RUN FUNCTION
def run_algorithm(name):

    if name == "GA":
        return run_ga(
            grid,
            pop_size=pop_size,
            num_sensors=sensors,
            generations=generations,
            selection_method=selection_method,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            diversity_type=diversity_type,
            survivor_method=survivor_method,
            init_method=init_method,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )

    if name == "DE":
        return run_de(
            grid,
            pop_size=pop_size,
            num_sensors=sensors,
            generations=generations
        )

    ga = run_algorithm("GA")
    de = run_de(grid, pop_size, sensors, generations)

    return ga if ga[0] > de[0] else de

# RESULTS
st.markdown("---")
st.markdown("## 📊 Results")

# SINGLE RUN
if mode == "Single Run":

    if st.button("🚀 Run Optimization"):

        best_fitness, history, avg_history, best_solution, diversity_history, snapshots = run_algorithm(single_algo)

        cov = coverage_ratio(best_solution, grid, radius)

        st.session_state.runs_history.insert(0, {
            "algo": single_algo,
            "fitness": best_fitness,
            "coverage": cov,
            "sensors": len(best_solution),
            "solution": best_solution,
            "history": history,
            "avg_history": avg_history,
            "diversity": diversity_history
        })

        st.success("✅ Run Completed")

# COMPARE MODE
else:

    if st.button("🚀 Run Full Comparison"):

        results = {}

        for name in ["GA", "DE", "HYBRID"]:

            best_fitness, history, avg_history, best_solution, diversity_history, snapshots = run_algorithm(name)

            cov = coverage_ratio(best_solution, grid, radius)

            results[name] = {
                "fitness": best_fitness,
                "coverage": cov,
                "sensors": len(best_solution),
                "solution": best_solution,
                "history": history,
                "avg_history": avg_history,
                "diversity": diversity_history
            }

        winner = max(results, key=lambda x: results[x]["fitness"])

        for name in ["GA", "DE", "HYBRID"]:
            st.session_state.runs_history.insert(0, {
                "algo": name,
                "fitness": results[name]["fitness"],
                "coverage": results[name]["coverage"],
                "sensors": results[name]["sensors"],
                "solution": results[name]["solution"],
                "history": results[name]["history"],
                "avg_history": results[name]["avg_history"],
                "diversity": results[name]["diversity"]
            })

        st.success(f"🏁 Winner: {winner}")

# HISTORY + REPORT
st.markdown("### 🧾 Runs History (Latest First)")

for idx, run in enumerate(st.session_state.runs_history):

    st.markdown(f"## 🔁 Run #{idx + 1} - {run['algo']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fitness", f"{run['fitness']:.4f}")
    col2.metric("Coverage", f"{run['coverage']:.4f}")
    col3.metric("Sensors", run["sensors"])

    st.markdown("### 📄 Full Report")

    avg_fitness = np.mean(run["history"])
    best_gen = int(np.argmax(run["history"]))

    st.write(f"""
    🔹 Algorithm: {run['algo']}  
    🔹 Best Fitness: {run['fitness']:.4f}
    🔹 Avg Fitness: {avg_fitness:.4f}
    🔹 Coverage: {run['coverage']:.4f}
    🔹 Sensors: {run['sensors']}
    🔹 Best Generation: {best_gen}
    """)

    st.markdown("#### 📈 Fitness Curve")
    plot_fitness(run["history"])

    st.markdown("#### 📉 Avg Fitness")
    fig, ax = plt.subplots()
    ax.plot(run["avg_history"])
    st.pyplot(fig)

    st.markdown("#### 🧬 Diversity")
    fig, ax = plt.subplots()
    ax.plot(run["diversity"])
    st.pyplot(fig)

    st.markdown("#### 🗺️ Sensor Map")

    sx = [x for x, y in run["solution"]]
    sy = [y for x, y in run["solution"]]

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="Blues", alpha=0.5, origin="lower")
    ax.scatter(sx, sy, c='red', s=80)

    st.pyplot(fig)

    st.markdown("---")

# FOOTER
st.markdown("---")
st.caption("EA Project - GA vs DE vs HYBRID Optimization Dashboard 🚀")