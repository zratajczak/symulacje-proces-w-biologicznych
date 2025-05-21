import numpy as np
import matplotlib.pyplot as plt

#Runge-Kutta 4
def runge_kutta(f, t, y, h, scenario_id, params):
    k1 = f(t, y, scenario_id, params)
    k2 = f(t + h / 2, y + h / 2 * k1, scenario_id, params)
    k3 = f(t + h / 2, y + h / 2 * k2, scenario_id, params)
    k4 = f(t + h, y + h * k3, scenario_id, params)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

#Model biologiczny
def biological_model(t, y, scenario_id, p):
    p53, MDM2_cyt, MDM2_nuc, PTEN = y

    p1, p2, p3, d1, d2, d3, k1, k2, k3 = p.copy()

    if scenario_id == 'B':
        p1 *= 2  #Uszkodzenie DNA
    elif scenario_id == 'C':
        p3 = 0  #Wyłączony PTEN
    elif scenario_id == 'D':
        p3 = 0  #Wyłączony PTEN
        k1 *= 0.1  # siRNA

    dp53 = p1 - d1 * p53
    dMDM2_cyt = k1 * (p2 * p53 / (k2 + p53)) - d2 * MDM2_cyt
    dMDM2_nuc = dMDM2_cyt * 0.5 - d2 * MDM2_nuc
    dPTEN = p3 - d3 * PTEN - k3 * PTEN * p53
    return np.array([dp53, dMDM2_cyt, dMDM2_nuc, dPTEN])

#Symulacja jednego scenariusza
def simulate_scenario(scenario_id, model, params):
    h = 1
    t_end = 48 * 60
    steps = int(t_end / h)
    y = np.array([100, 100, 50, 200])
    t = 0
    t_values = [t]
    y_values = [y.copy()]
    for _ in range(steps):
        y = runge_kutta(model, t, y, h, scenario_id, params)
        t += h
        t_values.append(t)
        y_values.append(y.copy())
    return np.array(t_values), np.array(y_values)

base_params = [
    8.8,  #p1
    440,  #p2
    400,  #p3
    1.375e-4,  #d1
    1.375e-4,  #d2
    3e-5,  #d3
    1.425e-4,  #k1
    0.5,  #k2
    1.5e-5  #k3
]
variable_names = ["p53", "MDM2_cyt", "MDM2_nuc", "PTEN"]

scenario_descriptions = {
    'A': "Brak uszkodzeń DNA, działa pętla PTEN, brak siRNA",
    'B': "Uszkodzenie DNA (komórki zdrowe)",
    'C': "Nowotwór – wyłączony PTEN, uszkodzone DNA",
    'D': "Terapia – wyłączony PTEN, uszkodzone DNA, obecne siRNA"
}

scenarios = ['A', 'B', 'C', 'D']
for scenario in scenarios:
    t_vals, y_vals = simulate_scenario(scenario, biological_model, base_params)

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(t_vals, y_vals[:, i], label=variable_names[i])
    plt.title(f"Scenariusz {scenario}: {scenario_descriptions[scenario]}")
    plt.xlabel("Czas (minuty)")
    plt.ylabel("Poziom zmiennych")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
