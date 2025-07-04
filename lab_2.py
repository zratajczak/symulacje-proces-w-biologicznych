import numpy as np
import matplotlib.pyplot as plt
import json
import os
import random
from scipy.interpolate import interp1d
import sympy as sp
import sys

# wczytanie parametrów i modelu
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

with open(os.path.join(base_path, "params.json")) as f:
    param_values = json.load(f)

with open(os.path.join(base_path, "model.json")) as f:
    model_data = json.load(f)

species_names = model_data["species"]
initial_state = model_data["initial_state"]
model_stoichiometry = model_data["model_matrix"]
rate_laws_str = model_data["rate_laws"]

# przygotowanie symboliczne sympy
species_syms = sp.symbols(species_names)
param_syms = sp.symbols(list(param_values.keys()) + ["p2_eff", "p3_eff", "d2_eff"])
local_syms = {s.name: s for s in species_syms + param_syms}

rate_expressions = [sp.sympify(expr, locals=local_syms) for expr in rate_laws_str]
rate_functions = [sp.lambdify(species_syms + param_syms, expr, modules='numpy') for expr in rate_expressions]

# funkcja obliczająca szybkości reakcji
def compute_rates(state, scenario_flags):
    p2_eff = param_values["p2"] * 0.02 if scenario_flags["siRNA"] else param_values["p2"]
    p3_eff = 0.0 if scenario_flags["pten_off"] else param_values["p3"]
    d2_eff = param_values["d2"] * 0.1 if not scenario_flags["dna_damage"] else param_values["d2"]

    args = (
        state +
        [param_values[k] for k in param_values] +
        [p2_eff, p3_eff, d2_eff]
    )
    return [f(*args) for f in rate_functions]

# parametry symulacji
t_end = 96 * 60  # minuty
dt = 1
time_points = np.arange(0, t_end + dt, dt)
num_runs = 3  # liczba realizacji do pokazania na wykresie

# etykiety i kolory
plot_labels = ["p53", "MDM2_cyt", "MDM2_nuc", "PTEN"]
plot_colors = ["blue", "green", "red", "orange"]

# scenariusze
scenarios = {
    "A_Zdrowa": {"siRNA": False, "pten_off": False, "dna_damage": False},
    "B_UszkodzenieDNA": {"siRNA": False, "pten_off": False, "dna_damage": True},
    "C_Nowotwor": {"siRNA": False, "pten_off": True, "dna_damage": True},
    "D_Terapia": {"siRNA": True, "pten_off": True, "dna_damage": True},
}

# Gillespie
def gillespie_alg(stoich_matrix, initial, rate_fn, t_end, scenario_flags):
    state = initial.copy()
    t = 0
    t_list = [0]
    state_list = [state.copy()]

    while t < t_end:
        rates = rate_fn(state, scenario_flags)
        total_rate = sum(rates)
        if total_rate <= 0:
            break

        r1, r2 = random.random(), random.random()
        tau = (1.0 / total_rate) * np.log(1.0 / r1)
        cumulative_rates = np.cumsum(rates)
        reaction_idx = np.searchsorted(cumulative_rates, r2 * total_rate)

        for i in range(len(state)):
            state[i] += stoich_matrix[reaction_idx][i]

        t += tau
        t_list.append(t)
        state_list.append(state.copy())

    return np.array(t_list), np.array(state_list)

# Next Reaction
def next_reaction(stoich_matrix, initial, rate_fn, t_end, scenario_flags):
    state = initial.copy()
    t = 0
    t_list = [0]
    state_list = [state.copy()]

    while t < t_end:
        rates = rate_fn(state, scenario_flags)
        if all(rate <= 0 for rate in rates):
            break

        taus = [np.random.exponential(1 / rate) if rate > 0 else np.inf for rate in rates]
        tau = min(taus)
        reaction_idx = taus.index(tau)

        for i in range(len(state)):
            state[i] += stoich_matrix[reaction_idx][i]

        t += tau
        t_list.append(t)
        state_list.append(state.copy())

    return np.array(t_list), np.array(state_list)

# przygotowanie folderu
output_dir = "wykres"
os.makedirs(output_dir, exist_ok=True)

# symulacja i rysowanie wykresów dla każdego scenariusza
for scenario_label, scenario_flags in scenarios.items():
    scenario_code = scenario_label.split("_")[0]
    print(f"\nUruchamianie scenariusza: {scenario_label}")

    # Gillespie
    plt.figure(figsize=(10, 7))
    for run in range(num_runs):
        t_sim, state_sim = gillespie_alg(model_stoichiometry, initial_state, compute_rates, t_end, scenario_flags)
        interpolators = [
            interp1d(t_sim, state_sim[:, i], kind='previous', bounds_error=False, fill_value='extrapolate')
            for i in range(4)
        ]
        interpolated = np.array([interp(time_points) for interp in interpolators]).T
        for i in range(4):
            plt.plot(
                time_points / 60,
                interpolated[:, i],
                color=plot_colors[i],
                alpha=0.5,
                label=plot_labels[i] if run == 0 else None
            )
    plt.title(f"Metoda Gillespiego – {scenario_label}")
    plt.xlabel("Czas (godziny)")
    plt.ylabel("Liczba cząsteczek")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path_gillespie = os.path.join(output_dir, f"gillespie_wykres_{scenario_code}.png")
    plt.savefig(save_path_gillespie)
    plt.close()
    print(f" Symulacja metodą Gillespiego - zapisano: {save_path_gillespie}")

    # Next Reaction
    plt.figure(figsize=(10, 7))
    for run in range(num_runs):
        t_sim, state_sim = next_reaction(model_stoichiometry, initial_state, compute_rates, t_end, scenario_flags)
        interpolators = [
            interp1d(t_sim, state_sim[:, i], kind='previous', bounds_error=False, fill_value='extrapolate')
            for i in range(4)
        ]
        interpolated = np.array([interp(time_points) for interp in interpolators]).T
        for i in range(4):
            plt.plot(
                time_points / 60,
                interpolated[:, i],
                color=plot_colors[i],
                alpha=0.5,
                label=plot_labels[i] if run == 0 else None
            )
    plt.title(f"Next Reaction – {scenario_label}")
    plt.xlabel("Czas (godziny)")
    plt.ylabel("Liczba cząsteczek")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path_next = os.path.join(output_dir, f"next_reaction_{scenario_code}.png")
    plt.savefig(save_path_next)
    plt.close()
    print(f" Symulacja metodą Next Reaction - zapisano: {save_path_next}")

print("\nSymulacje zakończone.")
