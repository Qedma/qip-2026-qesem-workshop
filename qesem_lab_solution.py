# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Characterization-based Error Mitigation for Quantum Computation - Workshop

# %% [markdown]
# ### Abstract
# In this lab, you will learn about **Quantum Error Suppression and Error Mitigation (QESEM)** through hands-on experiments.
# With QESEM, users can run quantum circuits on noisy QPUs and obtain highly accurate, error-free results with minimal QPU time overhead, close to theoretical limits.
# When executing a circuit, QESEM runs a device characterization protocol tailored to your circuit, yielding a reliable noise model for the errors occurring in the circuit. Based on the characterization, QESEM first implements noise-aware transpilation to map the input circuit onto a set of physical qubits and gates, which minimizes the noise affecting the target observable. These include the natively available gates (CX/CZ on IBM® devices), as well as additional gates optimized by QESEM, forming QESEM’s extended gate set. QESEM then runs a set of characterization-based error suppression (ES) and error mitigation (EM) circuits on the QPU and collects their measurement outcomes. These are then classically post-processed to provide an unbiased expectation value and an error bar for each observable, corresponding to the requested accuracy.
#
# For a detailed description of the QESEM algorithm, see our recent [paper](https://arxiv.org/abs/2508.10997) and the [QESEM documentation](https://docs.qedma.io/).
#
# <div>
# <img src="Qesem_workflow.svg" width="800"/>
# </div>

# %% [markdown]
# ## Setup
# We will run the workshop using google colab.
#
# Instead, if you prefer, you can clone the [repository](https://github.com/Qedma/ieee2025-qesem-workshop) and install the required packages locally using `pip` or `poetry`.
#
# Use the following commands to clone the repository and install the required packages.

# %%
# !git clone https://github.com/Qedma/ieee2025-qesem-workshop.git

# %%
# !pip install "qiskit>=2.0.0" "qiskit-ibm-runtime>=0.40.0" "qiskit-aer>=0.17.1" "networkx>=3.5" "matplotlib==3.10.0" "tqdm>=4.67.1" "scipy" "numpy" "qedma-api==0.18.3"

# In case you are running locally, please install jupyter as well:
# pip install "jupyter>=1.1.1"

# %%
# cd /content/ieee2025-qesem-workshop

# %% [markdown]
# # Imports

# %%
# %matplotlib inline

import datetime
import math
import time

import grader
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import qedma_api
import qiskit
import qiskit.converters
import qiskit.quantum_info
import qiskit.result
import qiskit_aer
import qiskit_aer.noise
import qiskit_ibm_runtime.fake_provider
import scipy.optimize
import tqdm
import utils

# %% [markdown]
# ## Learning Objectives

# %% [markdown]
# We'll explore:
#
# 1. **Why do we need error mitigation?** - See how quantum noise affects different types of measurements
# 2. **Kicked Ising circuits** - Our main example is the Kicked Ising chain, a benchmark model in quantum dynamics
# 3. **Using QESEM** - Use QESEM to mitigate the noise

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Table of Contents
#
# 1. [Section 1: Understanding the Importance of Error Mitigation](#Section-1:-Understanding-the-Importance-of-Error-Mitigation)
#    - [1.1 Use Case: Kicked Ising](#1.1-Use-Case:-Kicked-Ising)
#    - [1.2 Exercise 1: Comparing Ideal and Noisy Values](#1.2-Exercise-1:-Comparing-Ideal-and-Noisy-Values)
#      <!-- - [a. Trotter Circuit Visualization](#exercise-1a)
#      - [b. Ideal VS Noisy](#exercise-1b)
#        - [i. Total Magnetization](#exercise-1b-i)
#        - [ii. Heavy Weight (zzz)](#exercise-1b-ii) -->
# 2. [Section 2: Introduction to QESEM](#Section-2:-Introduction-to-QESEM)
#    - [QESEM Parameters Reference](#QESEM-Parameters-Reference)
#    - [2.1 Exercise 2: Applying QESEM on a simple circuit](#2.1-Exercise-2:-Applying-QESEM-on-a-simple-circuit)
#      <!-- - [a. Time Estimation (Analytical + Empirical)](#exercise-2a)
#      - [b. Execution](#exercise-2b)
#      - [c. Reading Results](#exercise-2c)
#      - [d. Execution Metrics](#exercise-2d) -->
#    - [2.2 Key Concepts](#2.2-Key-Concepts)
#      - [2.2.1 Volume and Active Volume](#2.2.1-Volume-and-Active-Volume)
#      - [2.2.2 QESEM Runtime Overhead](#2.2.2-QESEM-Runtime-Overhead)
#    - [2.3 Exercise 3: QPU time vs. Active Volume](#2.3-Exercise-3:-QPU-time-vs.-Active-Volume)
#      <!-- - [a. Observables vs QPU Time](#exercise-3a)
#      - [b. Influence of Angles](#exercise-3b) -->
#    - [2.4 Exercise 4: Exploring the $T\propto\frac{1}{\varepsilon^2}$ Relationship](#2.4-Exercise-4:-Exploring-the-$T-\propto-\frac{1}{\varepsilon^2}$-Relationship)
#      <!-- - [a. Mitigation Overhead vs 𝜖](#exercise-4a) -->

# %% [markdown]
# # Section 1: Understanding the Importance of Error Mitigation

# %% [markdown]
# Quantum computers are inherently noisy devices. Even small amounts of noise can significantly affect quantum computations, especially as circuit depth increases. In this section, we'll:
#
# 1. **Build a circuit** for the Kicked Ising model
# 2. **Learn the effects of noise** on different types of quantum observables

# %% [markdown]
# ## 1.1 Use Case: Kicked Ising
#
# The **Kicked Ising Model** is a paradigmatic quantum spin chain model used to study quantum chaos, entanglement, and non-equilibrium dynamics. It consists of spin-1/2 particles (qubits) with nearest-neighbor interactions on a graph, and a periodically applied "kick".
#
# ### Model Definition
#
# The time evolution of the kicked Ising model is governed by a periodically time-dependent Hamiltonian. The unitary evolution over one period $T$, on a 1D lattice of size $n$, is given by:
#
# $$
# U = \exp\left(-i J \sum_{j=0}^{n-2} \sigma_j^z \sigma_{j+1}^z \right) \exp\left(-i h_x \sum_{j=0}^{n-1} \sigma_j^x \right)
# $$
#
# where $\sigma_j^x, \sigma_j^z$ are Pauli matrices acting on site $j$, and $J, h_x$ are constants.
#
# ### Physical Significance
#
# - **Quantum Chaos:** The kicked Ising model exhibits rich quantum chaotic behavior, making it a benchmark for studies of thermalization and information scrambling in many-body quantum systems.
# - **Floquet Systems:** Since the system is driven periodically, it is a canonical example of a **Floquet system**, where stroboscopic (periodic) dynamics are studied.
# - **Entanglement & Quantum Information:** The model is widely used to investigate entanglement growth, operator spreading, and out-of-time-ordered correlators (OTOCs).
#
# ### Applications
#
# - Modeling non-equilibrium quantum dynamics.
# - Studying quantum information scrambling and entanglement spreading.
# - Exploring the transition between integrability and chaos in quantum systems.
#
# ---
#
# **References:**
# - Prosen, T. (2007). "Chaos and complexity in a kicked Ising spin chain." *Phys. Rev. E* 65, 036208.
# - [Wikipedia: Kicked Ising Model](https://en.wikipedia.org/wiki/Kicked_Ising_model)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Note:</b>
#
# While in this lab we focus on exploring a 1D Kicked Ising model for simplicity, QESEM can work on arbitrary circuits and observables.

# %% [markdown]
# ## 1.2 Exercise 1: Comparing Ideal and Noisy Values

# %% [markdown]
# This exercise demonstrates how quantum noise degrades computation results by comparing three scenarios:
#
# **What We'll Do:**
# - **Build Kicked Ising Circuits**: Create quantum circuits simulating time evolution with increasing number of Trotter steps
# - **Measure Observables**:
#   - **Z Correlation Operators**: Evaluate  $\left\langle Z_0 Z_1 \dots Z_{n-1} \right\rangle$ and $\left\langle Z_0 Z_1 \dots Z_{4} \right\rangle$ to probe global multi-qubit correlations
# - **Compare Results**: Ideal expectation values to noisy ones
#

# %% [markdown]
# ### Step 1: Preparing Kicked-Ising circuit

# %% [markdown] vscode={"languageId": "raw"}
# utils.kicked_ising_1d: Creates a quantum circuit that implements the Kicked Ising model. It applies alternating layers of RX gates (transverse field) and RZZ gates (interaction terms) to simulate the time evolution of the system.


# %%
def kicked_ising_1d(num_qubits: int, theta_x: float, theta_zz: float, num_steps: int) -> qiskit.QuantumCircuit:
    """
    Parameters:
        num_qubits (int): number of qubits on chain.
        theta_x (float): Angle for RX gates.
        theta_zz (float): Angle for RZZ gates.
        num_steps (int): Number of steps.

    Returns:
        QuantumCircuit: The resulting quantum circuit.
    """
    graph = nx.path_graph(num_qubits)
    qc = qiskit.QuantumCircuit(num_qubits)

    # Precompute edge layers (alternating non-overlapping pairs)
    edges = list(graph.edges())
    even_edges = [(u, v) for (u, v) in edges if u % 2 == 0]
    odd_edges = [(u, v) for (u, v) in edges if u % 2 == 1]

    for step in range(num_steps):
        # RX on all qubits
        for q in range(num_qubits):
            qc.rx(theta_x, q)

        # Apply even and odd layers separately
        for edge_layer in [even_edges, odd_edges]:
            for u, v in edge_layer:
                qc.rzz(theta_zz, u, v)

        if step < num_steps - 1:
            qc.barrier()

    return qc


# %% [markdown]
# ### Step 2: Kicked Ising Circuit Visualization

# %% [markdown] vscode={"languageId": "raw"}
# Here we set up the circuit experiment parameters and create a visualization of the circuit.
#

# %%
n_qubits_ex1 = 20
n_steps = 3

circ = kicked_ising_1d(n_qubits_ex1, theta_x=math.pi / 6, theta_zz=math.pi / 3, num_steps=n_steps)

print(f"Circuit 2q layers: {circ.depth(filter_function=lambda instr: len(instr.qubits) == 2)}")
print("\nCircuit structure:")

circ.draw("text", scale=0.8, fold=-1)


# %% [markdown]
# ### Step 3: Simulation Parameters
#
# Here we define:
#
# 1. A list of circuits to simulate (corresponding to the different number of time steps)
# 2. A list of observables to measure: $\langle Z_0...Z_4 \rangle$ and $\langle Z_0...Z_{n-1} \rangle$ and their labels
#

# %%
steps_range = range(1, 9)

circs_ex1 = []
for n_steps in steps_range:
    circs_ex1.append(
        kicked_ising_1d(
            n_qubits_ex1,
            theta_x=math.pi * 0.14,
            theta_zz=math.pi * 0.05,
            num_steps=n_steps,
        )
    )

# Prepare pairs of  (observables , labels)
observable_label_pairs = [
    (qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z" * 5, range(5), 1)], n_qubits_ex1), r"$Z_0Z_1...Z_{4}$"),
    (qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z" * n_qubits_ex1, range(n_qubits_ex1), 1)], n_qubits_ex1), r"$Z_0Z_1...Z_{n-1}$"),
]

# %% [markdown]
# ### Step 4: Compute ideal values

# %% [markdown]
# Here we compute the ideal expectation values for each observable using exact simulation with statevectors. <br>
# This gives us the theoretical values that we would obtain on a perfect quantum computer without any noise.
#
# The following dictionary holds the ideal/noisy expectation values for every observable per step:<br>
# graphs["ideal" or "noisy"][observable label] = [Expectation of observable at steps_range[0], Expectation of observable at steps_range[1], ....]

# %% [markdown]
# <div class="alert alert-block alert-success">
# <b>Exercise 1.1:</b> <br>
#     Complete the code for computing the ideal expectation value of "obs" after running the circuit "circ"
#
# Hint: [Statevector Documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector)

# %%
graphs = {"ideal": {}}
graphs["ideal"]["steps_range"] = steps_range
for circ in tqdm.tqdm(circs_ex1):  # Loop over circuits [one step circuit, two steps circuit ,....]
    for obs, label in observable_label_pairs:  # Loop over observables and their label
        if label not in graphs["ideal"]:  # Create list of ideal expectation values for every circuit
            graphs["ideal"][label] = []
        ideal_value = qiskit.quantum_info.Statevector(circ).expectation_value(obs).real
        graphs["ideal"][label].append(ideal_value)


# %%
# Function to draw graphs from "graphs" object

# Running the function
utils.graph_plots(graphs, observable_label_pairs)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Build Intuition:</b>
#
# Which of the two observables will be more affected by noise?

# %% [markdown]
# ### Step 5: Compute noisy values with error bars

# %% [markdown]
# Let's add **realistic quantum noise** to our simulation. We'll use an AerSimulator based on IBM's Fake Fez backend, which includes realistic noise models based on actual quantum hardware. <br>

# %%
fake_backend = qiskit_ibm_runtime.fake_provider.FakeFez()
basis_gates = fake_backend.configuration().basis_gates

noisy_backend = qiskit_aer.AerSimulator(
    method="matrix_product_state",
    noise_model=qiskit_aer.noise.NoiseModel.from_backend(fake_backend),
    basis_gates=basis_gates,
    coupling_map=fake_backend.configuration().coupling_map,
    properties=fake_backend.properties(),
    device="GPU" if "GPU" in qiskit_aer.AerSimulator().available_devices() else "CPU",
)


num_shots = 100  #### <------ Edit if too slow


# %% [markdown]
# Let's compute the noisy expectation values by running the circuits on the noisy simulator with finite shots. We calculate both the expectation values and their standard deviations to understand the statistical uncertainty in our measurements.

# %% [markdown]
# <div class="alert alert-block alert-success">
# <b>Exercise 1.2:</b> <br>
#     Complete the code for running "num_shots" shots of "transpiled_circ" on "noisy_backend" and extract the "counts" dictionary from the result. <br>
# Change the number of shots if the simulation is too slow.<br>
#
# Hint: [AerSimulator Documentation](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html)

# %%
graphs["noisy"] = {"steps_range": steps_range}
graphs["noisy_std"] = {}

for circ in tqdm.tqdm(circs_ex1):
    transpiled_circ = qiskit.transpile(circ, basis_gates=basis_gates, coupling_map=fake_backend.coupling_map)
    transpiled_circ = utils.remove_idle_qubits(transpiled_circ)

    transpiled_circ.measure_all()
    counts = noisy_backend.run(transpiled_circ, shots=num_shots).result().get_counts()

    for obs, label in observable_label_pairs:
        if label not in graphs["noisy"]:
            graphs["noisy"][label] = []
            graphs["noisy_std"][label] = []

        noisy_obs_expectation = qiskit.result.sampled_expectation_value(counts, obs)  # Convert counts to expectation value
        noisy_obs_variance = (1 - qiskit.result.sampled_expectation_value(counts, obs) ** 2) / num_shots  # Only true for Paulis: P^2=I
        graphs["noisy"][label].append(noisy_obs_expectation)
        graphs["noisy_std"][label].append(noisy_obs_variance**0.5)


# %% [markdown]
# ### Step 6: Plot the results

# %% [markdown]
# Finally, we visualize the results by plotting both the ideal and noisy expectation values as a function of the number of Trotter steps.

# %%
utils.graph_plots(graphs, observable_label_pairs)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Build intuition:</b> <br>
#
#
# - As circuits get deeper, quantum noise accumulates and the gap between ideal and noisy values grows. QESEM allows to run deeper circuits and still get meaningful results.
# - The decay of the noisy value is controlled by the number of noisy operations the observable is sensitive to. The $\langle Z0...Z19 \rangle$ observable decays much more than $\langle Z0...Z4 \rangle$ because its lightcone spans the entire circuit.

# %% [markdown]
# # Section 2: Introduction to QESEM

# %% [markdown]
# Now that we've seen how dramatically noise affects quantum computations, let's explore **QESEM** - a powerful technique to reduce these errors.
#
# ### How QESEM Works
#
# QESEM combines two approaches:
#
# 1. **Error Suppression**: Reduce the unitary part of the noise. It is based on QESEM's characterization.
# 2. **Error Mitigation**: Use classical post-processing to estimate what the results would have been without noise.

# %% [markdown]
# ### QESEM Parameters Reference
# During this section we will guide you through the main parameters and explain each one.
# There are more advanced parameters which we won't cover, see an elaborate explanation about all the parameters [here](https://docs.qedma.io/advanced_settings/).

# %% [markdown]
# ## 2.1 Exercise 2: Applying QESEM on a simple circuit

# %% [markdown]
# ### Step 1: Initialize qedma_api client

# %% [markdown]
# Here we are initializing the qedma_api client for an IBMQ simulator backend.
# Note: this API token will be available during the IEEE Quantum Week 2025 for you to explore QESEM on IBMQ simulators. For using QESEM on real QPUs please contact us at Qedma.

# %%
# configuration
# todo add your token here
qedma_api_token =
ibm_instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/test::"

qedma_client = qedma_api.Client(api_token=qedma_api_token)
provider = qedma_api.IBMQProvider(instance=ibm_instance)
qedma_client.set_provider(provider)

# %%
# please enter a job description prefix for you jobs, e.g. your name
job_description_prefix = "job_description_prefix"

# %% [markdown]
# ### Step 2: Prepare Circuit and Observables
# The following circuit prepares the state $|\psi\rangle = 0.6|0000\rangle+0.8|1111\rangle $
#
# We will measure two observables:<br>
# avg_magnetization_ex2 = $\frac{1}{4} \sum_j Z_j$ <br>
# all_z_ex2 = $Z_0...Z_3$

# %%
circ_ex2 = qiskit.QuantumCircuit(4)
circ_ex2.ry(0.927 * 2, 0)
circ_ex2.cx(0, 1)
circ_ex2.cx(1, 2)
circ_ex2.cx(2, 3)

avg_magnetization_ex2 = qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z", [q], 1 / 4) for q in range(4)], num_qubits=4)

all_z_ex2 = qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z" * 4, range(4), 1)], 4)

obs_list_ex2 = [avg_magnetization_ex2, all_z_ex2]


circ_ex2.draw("text", scale=0.7)


# %% [markdown]
# ### Step 3: Empirical Time Estimation
#
# Users would typically want to know how much QPU time is required for their experiment.
# However, this is considered a hard problem for classical computers.<br>
# QESEM offers two modes of time estimation to inform users about the feasibility of their experiments:
# 1. Analytical time estimation - gives a very rough estimation and requires no QPU time. This can be used to test if a transpilation pass would potentially reduce the QPU time.
# 2. Empirical time estimation (demonstrated here) - gives a pretty good estimation and uses a few minutes of QPU time.
#
# In both cases, QESEM outputs the time estimation for reaching the required precision for <b>all</b> observables.

# %%
# Start a job for empirical time estimation


job = qedma_client.create_job(
    circuit=circ_ex2,
    observables=obs_list_ex2,
    empirical_time_estimation=True,  # "empirical" - gets actual time estimates without running full mitigation
    precision=0.1,  # Precision for all observables
    backend="fake_fez",  # E.g. "ibm_brisbane"
    description=f"{job_description_prefix}-ex2",
)
print(f"Your job ID is: {job.job_id}")


# %%
# Wait for the empirical time estimation to complete. This takes a 1-3 minutes
time_estimation = qedma_client.wait_for_time_estimation(job.job_id)

# %%
print(f"Empirical time estimation (sec): {time_estimation.seconds}")

# %% [markdown] vscode={"languageId": "raw"}
# ### Step 4: Use QESEM to estimate the expectation values

# %% [markdown]
# <div class="alert alert-block alert-success">
# <b>Exercise 2:</b> <br>
#     Use QESEM to estimate the expectation values of "avg_magnetization_ex2" and "all_z_ex2" observables on the state generated by circ_ex2.  <br>
#     Start the job for which you ran the empirical time estimation above and set the max QPU time to 15 minutes. <br>
#

# %%
job_id_ex2 = job.job_id  # Use the job created above
max_qpu_time_ex2 = datetime.timedelta(minutes=15)

# %% [markdown]
# ### Test parameters

# %%
message = grader.grade_ex2(job=job, job_id=job_id_ex2, max_qpu_time_ex2=max_qpu_time_ex2)
print(message)


# %%
# choose parameters for the start QESEM job:

qedma_client.start_job(
    job_id=job_id_ex2,
    max_qpu_time=max_qpu_time_ex2,
)


# %% [markdown]
# ### Step 5: Reading the results

# %%
job_res = qedma_client.wait_for_job_complete(  # Blocking - takes 3-5 minutes
    job_id=job.job_id,
)
for i, obs in enumerate(obs_list_ex2):
    print("-" * 10)
    print("Observable: " + ["Average Magnetization", "ZZZZ"][i])
    print(f"Ideal: {qiskit.quantum_info.Statevector(circ_ex2).expectation_value(obs).real}")
    print(f"Noisy: {job_res.noisy_results[i][1].value} \u00b1 {job_res.noisy_results[i][1].error_bar}")
    print(f"QESEM: {job_res.results[i][1].value} \u00b1 {job_res.results[i][1].error_bar}")


print("-" * 10)
# Some of the data gathered during a QESEM run.
print(f"Gate fidelities found: {job_res.execution_details.gate_fidelities}")


# %%
# Results as a graph
x = np.arange(2)  # Column position
width = 0.06

fig, ax = plt.subplots(figsize=(4, 2.5 * 1.5))

# Plot the bars side by side
ax.bar(
    x - width,
    [qiskit.quantum_info.Statevector(circ_ex2).expectation_value(obs).real for obs in [avg_magnetization_ex2, all_z_ex2]],
    width,
    label="Ideal",
)
ax.bar(
    x,
    [job_res.noisy_results[0][1].value, job_res.noisy_results[1][1].value],
    width,
    label="Noisy",
    yerr=[job_res.noisy_results[0][1].error_bar, job_res.noisy_results[1][1].error_bar],
    capsize=6,
)
ax.bar(
    x + width,
    [job_res.results[0][1].value, job_res.results[1][1].value],
    width,
    label="QESEM",
    yerr=[job_res.results[0][1].error_bar, job_res.results[1][1].error_bar],
    capsize=6,
)

ax.set_xticks(x)
ax.set_xticklabels(["Average Magnetization", "ZZZZ"])
ax.set_title(r"Comparing Ideal, Noisy and QESEM for $|\psi\rangle = 0.6|0000\rangle+0.8|1111\rangle$ ")
ax.legend()

plt.tight_layout()
plt.grid(axis="y")
plt.axhline(0, color="black", linewidth=2.5)  # y=0, bold black line
plt.show()


# %% [markdown]
# ## 2.2 Key Concepts

# %% [markdown]
# ### 2.2.1 Volume and Active Volume

# %% [markdown]
# A key figure of merit for quantifying the hardness of both error mitigation and classical simulation for a given circuit and observable is **active volume**: The number of two-qubit gates affecting the observable in the circuit. Different circuits and observables have different **active volumes**, which affects how hard they are to error-mitigate.
#
# The active volume depends on:
# - Circuit depth and width
# - Observable weight (number of non-identity Pauli operators)
# - Circuit structure (light cone of the observable)
#
# For further details, see the talk from the [2024 IBM Quantum Summit](https://www.youtube.com/watch?v=Hd-IGvuARfE&t=1730s&ab_channel=IBMResearch).
#
# <div>
# <img src="active_vol.svg" />
# </div>

# %% [markdown]
# ### 2.2.2 QESEM Runtime Overhead

# %% [markdown]
# The QPU time behaves roughly like:
# $$T_{QPU} \approx \left(\frac{A}{\epsilon^2}\right) \times e^{C \times IF \times V_{active}} + B$$
# Where:
# - B overhead of gate optimization and error characterization
# - precision $\epsilon$ absolute error in expectation value
# - $ IF \times V_{active} $ infidelity-per-gate times active volume. The active volume only includes gates within the active light-cone.
#
# The **overhead** of error mitigation tells us how many additional quantum resources we need to achieve a target precision.

# %% [markdown]
# ## 2.3 Exercise 2: QPU time vs. Active Volume
# In this section we'll fix the number of steps and the precision, and compare the QPU time for the two observables in Exercise 1.
#

# %% [markdown]
# <div class="alert alert-block alert-success">
# <b>Exercise 3:</b> <br>
#     Use QESEM's <b>analytical</b> time estimation for measuring the expectation of $\langle Z_0...Z_4\rangle$ and $\langle Z_0...Z_{n-1}\rangle$ of the state generated by a few steps of the Kicked Ising model.<br>
#     Set the default_precision to 0.02.
#
#

# %%
# Set the options parameter for the QESEM job by the instruction:

precision = 0.02
empirical_time_estimation = False


# %% [markdown]
# ### Test parameters

# %%
message = grader.grade_ex3(precision, empirical_time_estimation)
print(message)

# %% [markdown]
# ### Start a jobs only if parameter check passed

# %%
if message == grader.CORRECT_MESSAGE:
    print("Starting jobs")
    selected_step = 6  # number of steps in the circuit.
    num_qubits_ex3 = 20

    circ = kicked_ising_1d(
        num_qubits_ex3,
        theta_x=math.pi * 0.14,
        theta_zz=math.pi * 0.05,
        num_steps=selected_step,
    )

    # Prepare pairs of  (observables , labels)
    observable_label_pairs_ex3 = [
        (qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z" * 5, range(5), 1)], num_qubits_ex3), r"$Z_0Z_1...Z_{4}$"),
        (qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z" * num_qubits_ex3, range(num_qubits_ex3), 1)], num_qubits_ex3), r"$Z_0Z_1...Z_{n-1}$"),
    ]

    volume_jobs: list[datetime.timedelta] = []
    for observable, obs_label in observable_label_pairs_ex3:
        # run analytical time estimation on each observable separately. Takes about 3 minutes
        job = qedma_client.create_job(
            circuit=circ,
            observables=[observable],
            backend="fake_fez",
            precision=precision,
            empirical_time_estimation=empirical_time_estimation,
            description=f"{job_description_prefix}-ex3-{obs_label}",
        )
        time_est = qedma_client.wait_for_time_estimation(job.job_id)  # Wait for the time estimation to complete
        print(f"Analytical time estimation for observable {obs_label}: {time_est.seconds / 60} min")

else:
    print("Parameters check failed. Jobs did not start")


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Build intuition:</b> While the analytical time estimation is very rough (resolution of 30 minutes) and pessimistic, it still captures the difference in the active volumes of the two observables fairly quickly, and without using any QPU time.

# %% [markdown]
# ## 2.4 Exercise 4: Exploring the $T \propto \frac{1}{\varepsilon^2}$ Relationship

# %% [markdown]
# In this exercise, we will investigate **how QPU time scales with precision requirements**.
# ### Step 1: Create and visualize the test circuit
#
# We'll create a single Kicked Ising circuit to test the precision-time relationship. This circuit will have enough complexity to show clear scaling behavior while remaining manageable for the experiment.

# %% [markdown]
# <div class="alert alert-block alert-success">
#
# <b> Exercise 4.1:</b> <br>
# Complete the code. Use the function kicked_ising_1d() defined in Exercise 1 to create the circuit circ_ex4. Be sure to use the variables defined below n_qubits_ex4, n_steps_ex4, theta_x_ex4, theta_zz_ex4.
# </div>

# %%

n_qubits_ex4 = 5  # Number of qubits (enough to see scaling effects)
n_steps_ex4 = 10  # Number of Trotter steps (creates sufficient circuit depth)
theta_x_ex4 = math.pi / 6  # Rotation angle for X gates (transverse field strength)
theta_zz_ex4 = math.pi / 3  # Rotation angle for ZZ gates (interaction strength)

# Create the Kicked Ising circuit that will be used for precision testing
circ_ex4 = kicked_ising_1d(n_qubits_ex4, theta_x=theta_x_ex4, theta_zz=theta_zz_ex4, num_steps=n_steps_ex4)


# %%
message = grader.grade_ex_4_1(circ_ex4)
print(message)

# %%
print(f"Circuit 2q layers: {circ_ex4.depth(filter_function=lambda instr: len(instr.qubits) == 2)}")
print("\nCircuit structure:")

circ_ex4.draw("text", scale=1, fold=-1)

# %% [markdown]
# ### Step 2: Submit time estimation jobs with different precision values
#
# We'll submit multiple QESEM jobs, each with a different precision requirement (ε). The range from 0.005 to 0.06 covers both high-precision (expensive) and moderate-precision (cheaper) regimes.
#
# The observable: $\langle Z_0...Z_4 \rangle$. <br>
#
#
# **Empirical time estimation**: We will use "empirical" time estimation this setting performs a small execution on the QPU without running full mitigation. This usually takes a few minutes per estimation and is more reliable than the "analytical" estimation.

# %%
backend_name_fake = "fake_fez"
precisions_ex4 = [0.005, 0.01, 0.03, 0.05]
observable_ex4 = qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z" * n_qubits_ex4, range(n_qubits_ex4), 1)], n_qubits_ex4)  # Z_0..Z_{n-1}

# Initialize list to store job IDs
jobs_ex4 = []


# %%
if message == grader.CORRECT_MESSAGE:
    # Loop through each precision value
    for precision in precisions_ex4:
        print(f"Submitting job with precision: {precision:.3f}")

        job = qedma_client.create_job(
            circuit=circ_ex4,
            observables=[observable_ex4],
            precision=precision,
            backend=backend_name_fake,
            empirical_time_estimation=True,  # "empirical" - gets actual time estimates without running full mitigation
            description=f"{job_description_prefix}-ex4-{precision:.3f}",
        )

        # Store the job ID
        jobs_ex4.append(job)
        print(f"Job submitted with ID: {job.job_id}")

    print("\nAll jobs submitted!")

    start_time = time.time()
else:
    print("Parameter check failed. Jobs did not start")

# %% [markdown]
# ### Step 3: Monitor job progress
#
# Since time estimation jobs can take several minutes to complete, we'll monitor their progress. This gives us real-time feedback on when all jobs are finished. Might take approximately 15 minutes (tested on fake_fez).

# %%
print("Monitoring job completion...   (this takes around 15 minutes, tested on fake_fez, you may break and return later)")
print(f"Total jobs: {len(jobs_ex4)}")
print("-" * 30)

while True:  # Each job performs device characterization to estimate QPU time requirements
    # Count how many jobs are done
    done_count = 0
    for job in jobs_ex4:  # Jobs with smaller ε values may take longer to process
        if qedma_client.get_job(job.job_id).status == "ESTIMATED":
            done_count += 1

    # Show current status
    elapsed_time = time.time() - start_time
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {done_count}/{len(jobs_ex4)} jobs completed (elapsed: {elapsed_time / 60:.1f} min)")

    # Check if all jobs are done
    if done_count == len(jobs_ex4):
        print("\nAll jobs completed!")
        break

    # Wait 30 seconds before next check
    time.sleep(30)

end_time = time.time()
print(f"Total time: {(end_time - start_time)/60:.1f} minutes")

# %% [markdown] vscode={"languageId": "raw"}
# ### Step 4: Extract time estimation results
#
# Once all jobs are complete, we extract the QPU time estimates from each job's. These time estimates represent how long QESEM would need to achieve the requested precision.

# %%
# Once jobs are complete, extract time estimations
time_estimations = []

print("\nExtracting time estimations from jobs...")
for i, job in enumerate(jobs_ex4):  # Loop through all completed jobs
    print(f"Getting time estimation for job {i+1}")

    # Get the result and extract time estimation
    time_estimation_sec = qedma_client.wait_for_time_estimation(job.job_id).seconds

    # Store the time estimation
    time_estimations.append(time_estimation_sec)
    print(f"Time estimation: {time_estimation_sec} seconds")

print("\nAll time estimations extracted!")


# %% [markdown] vscode={"languageId": "raw"}
#
# ### Step 5: Visualize the results
#
# We'll create a standard linear plot to visualize how QPU time varies with precision. This gives us an intuitive view of the relationship.
#
# **Expected Pattern**: Time estimates should increase dramatically as precision requirements become more stringent (smaller ε).


# %%
# Define the plotting function to visualize QPU time vs precision
def plot_qpu_time_vs_precision(precisions: list[float], time_estimations_sec: list[float], title_suffix: str = "") -> None:
    """
    Plot QPU time estimation vs precision with curve fitting

    Args:
        precisions: Array of precision values (ε)
        time_estimations_sec: Array of time estimations in seconds
        title_suffix: Optional suffix to add to the plot title
    """
    # Convert time estimations from seconds to minutes
    time_estimations_min = np.array(time_estimations_sec) / 60

    # Define the fitting function: T = A/precision^2 + B
    def fit_function(precisions: list[float], a: float, b: float) -> list[float]:
        return [a / (p**2) + b for p in precisions]

    # Perform the curve fit using time
    popt, _ = scipy.optimize.curve_fit(fit_function, precisions, time_estimations_min)
    a_fit, b_fit = popt

    # Create a smooth curve for plotting the fit
    precision_smooth = np.linspace(min(precisions), max(precisions), 100).tolist()
    time_fit = fit_function(precision_smooth, a_fit, b_fit)

    plt.figure(figsize=(8, 5))
    plt.plot(precisions, time_estimations_min, "o", markersize=8, color="blue", label="Data")
    plt.plot(
        precision_smooth,
        time_fit,
        "-",
        linewidth=2,
        color="red",
        label=f"Fit: T = {a_fit:.3f}/ε² + {b_fit:.1f}",
    )
    plt.xlabel("Precision (ε)", fontsize=12)
    plt.ylabel("QPU Time Estimation (minutes)", fontsize=12)
    plt.title(f"QESEM QPU Time Estimation vs Precision{title_suffix}", fontsize=14)

    plt.locator_params(axis="y", nbins=10)  # Increase number of y-axis ticks
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Display fit parameters
    print("Fitted parameters:")
    print(f"A = {a_fit:.3f}")
    print(f"B = {b_fit:.2f}")
    print(f"Function: T = {a_fit:.3f}/ε² + {b_fit:.2f}")
    print(f"Data points: {len(precisions)}")

    plt.tight_layout()
    plt.show()


# Create the initial plot with Steps 2-5 data
plot_qpu_time_vs_precision(precisions_ex4, time_estimations)

# %% [markdown]
# And indeed we can see the expected functional dependence of:
# $$T_{QPU} \approx \frac{\tilde{A}}{\epsilon^2} + B$$
# Where:
# - $\tilde A={A} \times e^{C \times IF \times V_{active}}$
# - $A$ is a constant that depends on the circuit and the expectation value
# - $B$ is overhead due to gate optimization and error characterization
# - $C$ is a constant between 2-4 that depends on the details of the error mitigation method.

# %% [markdown]
# ### Step 6: Execute with error mitigation
#
# Now let's run the actual QESEM error mitigation on our circuit.

# %% [markdown]
# <div class="alert alert-block alert-success">
#
# <b> Exercise 4.2:</b>
#
# Look at the results of the QPU time vs precision graph above and think which precision you would like for your full execution (on a noisy simulator).
# Complete the QESEM job parameters accordingly.
# We will run an empirical time estimation for your desired precision to predict QPU resource requirements before running the full QESEM mitigation.
#
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Use a moderate precision for reasonable execution time while still demonstrating the mitigation capabilities.


# %%
precision_ex4_2 = 0.04
circuit_ex4_2 = circ_ex4
observables_ex4_2 = [observable_ex4]
backend_ex4_2 = "fake_fez"
empirical_time_estimation_ex4_2 = True
# %% [markdown]
# ### Test parameters

# %%
message = grader.grade_ex_4_2(
    precision_ex4_2=precision_ex4_2,
    circuit_ex4_2=circuit_ex4_2,
    observables_ex4_2=observables_ex4_2,
    backend_ex4_2=backend_ex4_2,
    empirical_time_estimation_ex4_2=empirical_time_estimation_ex4_2,
)
print(message)

# %% [markdown]
# ### Start a job only if parameter check passed

# %%
if message == grader.CORRECT_MESSAGE:  # Run the job only if parameters are correct
    # Start a job
    # Get time estimation first (quick empirical check)
    job_ex4_2 = qedma_client.create_job(
        circuit=circuit_ex4_2,
        observables=observables_ex4_2,
        precision=precision_ex4_2,
        backend=backend_ex4_2,
        empirical_time_estimation=empirical_time_estimation_ex4_2,
    )

    print(f"Time estimation job submitted: {job_ex4_2.job_id}")
    print(f"with precision: {precision_ex4_2}")

else:
    print(message)
    print("Parameter check failed. Job did not start.")


# %% [markdown]
# <a id="tips"></a>
# <div class="alert alert-block alert-info">
#
# <b> Tip:</b> Use this code cell to check your job's status.

# %%
qedma_client.get_job(job_ex4_2.job_id).status

# %% [markdown]
# Extract the time estimation from Step 6a and add it to our precision vs. QPU time graph to see how the new data point fits the T ∝ 1/ε² relationship.

# %%
# Get the time estimation result
time_est_result = qedma_client.wait_for_time_estimation(job_ex4_2.job_id)
step6_time_estimation = time_est_result.seconds

print(f"Step 6 time estimation: {step6_time_estimation} seconds")

# Add the new data point to existing arrays
precisions_updated = precisions_ex4 + [precision_ex4_2]
time_estimations_updated = time_estimations + [step6_time_estimation]

# Plot the updated graph with the new data point
print("\nUpdated graph with Step 6 data point:")
plot_qpu_time_vs_precision(precisions_updated, time_estimations_updated, " (Including Step 6)")


# %% [markdown]
# <div class="alert alert-block alert-success">
#
# <b> Exercise 4.3:</b>
#
# If you are satisfied with the time estimation go ahead and run the full QESEM error mitigation on our 5-qubit, 10-step Kicked Ising circuit with precision ε. In a case of real QPU, This will take QPU time according to your estimation result.
# Complete the QESEM job parameters accordingly.
#
# </div>

# %% [markdown]
# <a id="tips"></a>
# <div class="alert alert-block alert-info">
#
# <b> Tip:</b> Use "max_execution_time" option, this allows you to limit the QPU time, as `datetime.timedelta`. After the time limit is reached, QESEM stops sending new circuits. Circuits that have already been sent continue running, you can see detailed explanation [here](https://docs.qedma.io/advanced_settings/).
# Note: QESEM will end its run when it reaches the target precision or when it reaches max_execution_time, whichever comes first.
# We will define max execution time as 3 minutes, which is enough for the 5-qubit, 10-step Kicked Ising circuit with precision ε=0.04 according to our time estimation.

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Warning:</b> The QPU time estimation changes from one backend to another. Therefore, when executing QESEM, make sure to run it on the same backend that was selected when obtaining the QPU time estimation.

# %%
max_qpu_time_ex4_3 = datetime.timedelta(minutes=3)

# %% [markdown]
# ### Test parameters

# %%
message = grader.grade_ex_4_3(max_qpu_time_ex4_3=max_qpu_time_ex4_3)
print(message)

# %% [markdown]
# ### Start a job only if parameter check passed

# %%
if message == grader.CORRECT_MESSAGE:  # Run the job only if parameters are correct
    # Start a job
    # Submit actual mitigation job
    qedma_client.start_job(job_ex4_2.job_id, max_qpu_time=max_qpu_time_ex4_3)

    print(f"QESEM mitigation job submitted: {job_ex4_2.job_id}")
    print("Job is running... This may take several minutes.")

else:
    print(message)
    print("Parameter check failed, job did not start.")


# %% [markdown]
# <a id="tips"></a>
# <div class="alert alert-block alert-info">
#
# <b> Tip:</b> Use this code cell to check your job's status.

# %%
qedma_client.get_job(job_ex4_2.job_id).status

# %% [markdown]
# Extract and analyze the QESEM results, comparing ideal, noisy, and error-mitigated expectation values to demonstrate the effectiveness of QESEM.
#

# %%
# Read and analyze results
print("Reading QESEM results...")

# Get the mitigated results
qesem_job_results = qedma_client.wait_for_job_complete(job_ex4_2.job_id)
print("Job completed successfully!")

# Extract results for our observable
mitigated_expectation = qesem_job_results.results[0][1].value
mitigated_std = qesem_job_results.results[0][1].error_bar

# Get noisy results for comparison
noisy_expectation = qesem_job_results.noisy_results[0][1].value
noisy_std = qesem_job_results.noisy_results[0][1].error_bar

# Calculate ideal value for comparison
ideal_expectation = qiskit.quantum_info.Statevector(circ_ex4).expectation_value(observable_ex4).real

# Display comprehensive results summary
print("\n" + "=" * 60)
print("EXERCISE 4 - QESEM RESULTS SUMMARY")
print("=" * 60)
print(f"Observable: Global Z measurement (Z^⊗{n_qubits_ex4})")
print(f"Circuit: {n_steps_ex4}-step Kicked Ising, {n_qubits_ex4} qubits")
print(f"Precision target: {precision_ex4_2}")
print("-" * 60)
print(f"Ideal value:      {ideal_expectation:.6f}")
print(f"Noisy value:      {noisy_expectation:.6f} ± {noisy_std:.6f}")
print(f"QESEM value:      {mitigated_expectation:.6f} ± {mitigated_std:.6f}")
print("-" * 60)
print(f"Noisy error:      {abs(noisy_expectation - ideal_expectation):.6f}")
print(f"QESEM error:      {abs(mitigated_expectation - ideal_expectation):.6f}")
print(f"Error reduction:  {abs(noisy_expectation - ideal_expectation)/abs(mitigated_expectation - ideal_expectation if mitigated_expectation != ideal_expectation else 1):.1f}x")
print("-" * 60)
print(f"QESEM within target precision: {'✓' if abs(mitigated_expectation - ideal_expectation) <= precision_ex4_2 else '✗'}")
print("-" * 60)
# Additional detailed metrics
print(f"Total QPU time: \n {qesem_job_results.total_execution_time.seconds:.2f} seconds")
print(f"Gates fidelity measured during the experiment: \n {qesem_job_results.execution_details.gate_fidelities}")
print(f"Total shots / mitigation shots: \n {qesem_job_results.execution_details.total_shots} / {qesem_job_results.execution_details.mitigation_shots}")
print("=" * 60)


# %% [markdown]
# # Additional information
# If you have any questions or need further assistance, feel free to ask!
# In order to get access to QESEM over real QPUs, please contact us at Qedma.
#
# <b>Created by:</b> Yotam Lifshitz, Qedma. Email: Yotam.Lifshitz@qedm.com<br>
#
