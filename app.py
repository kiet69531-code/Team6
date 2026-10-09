
# ============================================================
# BOARD GAME WINNING PROBABILITIES (C4)
# Frontend: Streamlit
# Backend: NumPy + SymPy
# Mathematics: Markov Chains, Diagonalization, T^n
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- FRONTEND CONFIG ------------------------

st.set_page_config(
    page_title="Board Game Probability Lab",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Board Game Winning Probabilities")
st.caption(
    "B.Tech Mathematics Mini Project C4 | "
    "Markov Chains • Diagonalization • Steady State"
)

st.markdown("""
Explore how a game's transition probabilities affect its outcome.
Change the inputs and calculate winning probabilities dynamically.
""")

# -------------------- SIDEBAR INPUTS ------------------------

st.sidebar.header("⚙️ Game Configuration")

turns = st.sidebar.number_input(
    "Number of turns",
    min_value=1,
    max_value=1000,
    value=50,
    step=1
)

initial_state = st.sidebar.selectbox(
    "Starting state",
    ["Start", "Middle", "Final Stage"]
)

st.sidebar.info(
    "The benchmark is an illustrative 5-state game. "
    "Edit the transition matrix in the code to define other rules."
)

# -------------------- BACKEND: INPUT MODEL -------------------

states = [
    "Start",
    "Middle",
    "Final Stage",
    "Player A Wins",
    "Player B Wins"
]

transient_count = 3

# Rows = current state; columns = next state.
# This is an explicitly defined benchmark, not measured game data.
T = np.array([
    [0.2, 0.5, 0.3, 0.0, 0.0],
    [0.0, 0.3, 0.4, 0.2, 0.1],
    [0.0, 0.0, 0.4, 0.4, 0.2],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
], dtype=float)

# -------------------- BACKEND: VALIDATION --------------------

if not np.allclose(T.sum(axis=1), 1.0):
    st.error("Invalid transition matrix: each row must sum to 1.")
    st.stop()

if np.any(T < 0):
    st.error("Probabilities cannot be negative.")
    st.stop()

start_index = states.index(initial_state)

# Initial probability vector
p0 = np.zeros(len(states))
p0[start_index] = 1.0

# -------------------- BACKEND: MATRIX POWER ------------------

T_n = np.linalg.matrix_power(T, int(turns))
p_n = p0 @ T_n

# -------------------- BACKEND: DIAGONALIZATION ---------------

T_sym = sp.Matrix([
    [sp.Rational(str(x)) for x in row]
    for row in T
])

diagonalization_ok = True

try:
    P, D = T_sym.diagonalize()
    T_n_diag = P * (D ** int(turns)) * P.inv()

    # Verify the diagonalization result against direct power.
    assert T_n_diag == T_sym ** int(turns)

except (sp.MatrixError, ValueError, AssertionError):
    diagonalization_ok = False
    T_n_diag = None

# -------------------- BACKEND: ABSORBING CHAIN ---------------

Q = T[:transient_count, :transient_count]
R = T[:transient_count, transient_count:]
I = np.eye(transient_count)

try:
    N = np.linalg.inv(I - Q)

    # Eventual probabilities of reaching each winning state
    B = N @ R

    # Expected turns before reaching either winning state
    expected = N @ np.ones(transient_count)

except np.linalg.LinAlgError:
    st.error(
        "Could not compute expected turns. "
        "The fundamental matrix is singular."
    )
    st.stop()

win_probs = B[start_index]
expected_turns = expected[start_index]

# -------------------- FRONTEND: SUMMARY METRICS --------------

st.subheader("📊 Results Dashboard")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Player A — eventual win",
        f"{win_probs[0] * 100:.2f}%"
    )

with c2:
    st.metric(
        "Player B — eventual win",
        f"{win_probs[1] * 100:.2f}%"
    )

with c3:
    st.metric(
        "Expected turns",
        f"{expected_turns:.2f}"
    )

# -------------------- FRONTEND: MATRIX DISPLAY ---------------

st.subheader("🔢 Transition Matrix T")

st.write(
    "Each row represents the current state; "
    "each column represents the next state."
)

st.dataframe(
    pd.DataFrame(T, index=states, columns=states),
    use_container_width=True
)

# -------------------- FRONTEND: TURN DISTRIBUTION ------------

st.subheader(f"🎯 State Probabilities After {turns} Turns")

distribution = pd.DataFrame({
    "State": states,
    "Probability": p_n,
    "Percentage": p_n * 100
})

st.dataframe(
    distribution.style.format({
        "Probability": "{:.5f}",
        "Percentage": "{:.2f}%"
    }),
    use_container_width=True
)

unfinished = p_n[:transient_count].sum()

st.info(
    f"Probability the game is still in progress after "
    f"{turns} turns: {unfinished * 100:.2f}%"
)

# -------------------- FRONTEND: WINNING PROBABILITIES --------

st.subheader("🏆 Eventual Winning Probabilities")

winning_df = pd.DataFrame({
    "Winner": ["Player A", "Player B"],
    "Probability": win_probs
})

fig1, ax1 = plt.subplots(figsize=(7, 4))

sns.barplot(
    data=winning_df,
    x="Winner",
    y="Probability",
    hue="Winner",
    legend=False,
    ax=ax1
)

ax1.set_ylim(0, 1)
ax1.set_ylabel("Probability")
ax1.set_title("Eventual Chance of Winning")

for i, value in enumerate(win_probs):
    ax1.text(i, value + 0.02, f"{value:.1%}", ha="center")

st.pyplot(fig1)
plt.close(fig1)

# -------------------- FRONTEND: PROBABILITY OVER TIME --------

st.subheader("📈 Probability Evolution")

history = [p0.copy()]
current = p0.copy()

for _ in range(int(turns)):
    current = current @ T
    history.append(current.copy())

history = np.array(history)

fig2, ax2 = plt.subplots(figsize=(10, 4))

for i, state in enumerate(states):
    ax2.plot(
        range(int(turns) + 1),
        history[:, i],
        label=state,
        linewidth=2
    )

ax2.set_xlabel("Turn")
ax2.set_ylabel("Probability")
ax2.set_title("How State Probabilities Change")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="best", fontsize=8)

st.pyplot(fig2)
plt.close(fig2)

# -------------------- FRONTEND: HEATMAP ----------------------

st.subheader("🌡️ Transition Heatmap")

fig3, ax3 = plt.subplots(figsize=(9, 5))

sns.heatmap(
    T,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    xticklabels=states,
    yticklabels=states,
    ax=ax3
)

ax3.set_xlabel("Next State")
ax3.set_ylabel("Current State")
ax3.set_title("Transition Probabilities")

st.pyplot(fig3)
plt.close(fig3)

# -------------------- MATHEMATICS EXPLANATION ----------------

with st.expander("📚 Understand the mathematics"):
    st.markdown(r"""
    **1. Matrix powers**

    $$p_n = p_0 T^n$$

    The probability distribution after n turns is calculated
    by multiplying the initial distribution by the transition
    matrix raised to the nth power.

    **2. Diagonalization**

    $$T = P D P^{-1}$$

    $$T^n = P D^n P^{-1}$$

    This method works when T is diagonalizable.

    **3. Fundamental matrix**

    $$N = (I-Q)^{-1}$$

    **4. Eventual winning probabilities**

    $$B = NR$$

    **5. Expected turns until a winner**

    $$t = N\\mathbf{1}$$

    These absorbing-chain formulas require eventual absorption
    with finite expected duration.
    """)

if diagonalization_ok:
    st.success(
        f"Diagonalization verified for T^{int(turns)}."
    )
else:
    st.warning(
        "This matrix could not be diagonalized using the "
        "selected method. Direct matrix powers were used."
    )

# -------------------- DOWNLOAD RESULTS -----------------------

st.subheader("📥 Export Results")

csv_data = distribution.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download state probabilities as CSV",
    data=csv_data,
    file_name="board_game_probabilities.csv",
    mime="text/csv"
)

st.success("Analysis completed.")
