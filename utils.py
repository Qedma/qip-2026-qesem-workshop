from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import qiskit
import qiskit.converters
import qiskit.quantum_info


def graph_plots(
    graphs: dict[str, dict[str, Sequence[int | float]]],
    observable_label_pairs: list[tuple[qiskit.quantum_info.SparsePauliOp, str]],
) -> None:
    """
    Function to draw graphs from "graphs" object

    Input: graphs object (dictionary)
    Display graphs of noisy (optional) and ideal values vs step
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)

    j = 0  # axis index
    for _, obs_label in observable_label_pairs:
        ax = axs[j]
        ax.set_title("Kicked Ising Model - ideal vs noisy")
        ax.set_ylabel(r"$\langle$" + obs_label + r"$\rangle$")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # Makes x axis integers
        ax.set_xlabel("Steps")
        ax.plot(
            graphs["ideal"]["steps_range"],
            graphs["ideal"][obs_label],
            label="Ideal values",
            marker=".",
            color="black",
        )
        if "noisy" in graphs:  # Plot the noisy graph if it exists.
            ax.errorbar(
                graphs["noisy"]["steps_range"],
                graphs["noisy"][obs_label],
                np.array(graphs["noisy_std"][obs_label]) ** 0.5,
                label="Noisy values",
                marker=".",
                capsize=3,
            )
        ax.legend()
        ax.grid()
        j = j + 1

    plt.show()


def remove_idle_qubits(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """
    Removes qubits that are idle throughout the circuit
    """
    used = {q for op in qc for q in op.qubits}
    dag = qiskit.converters.circuit_to_dag(qc)
    for q in set(qc.qubits) - used:
        dag.remove_qubits(q)
    return qiskit.converters.dag_to_circuit(dag)
