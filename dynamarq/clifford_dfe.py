import random

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Clifford

"""
Code to construct state preparation and measurement qiskit circuits for performing a
direct fidelity estimation of Clifford circuits.

Implements the technique from:
    - Merkel et al., `When Clifford benchmarks are sufficient`, arXiv:2503.05943 (Sec. IV.A, IV.B)

"""

def sample_random_pauli(n, data_qubits=None):
    """ Sample a random n-qubit Pauli string that is not all identity. """
    if data_qubits is None :
        data_qubits = [1]*n
    assert any(data_qubit == 1 for data_qubit in data_qubits), \
            "Circuit must have at least one data qubit."
    letters = ['I', 'X', 'Y', 'Z']
    while True:
        s = ''.join(random.choice(letters) if data_qubits[i]==1 else 'I' for i in range(n))
        if any(ch != 'I' for ch in s):
            return s


def prep_eigenstate(pauli: Pauli) :
    """ Prepare +1 eigenstate of product-Pauli on |0>^n. """
    n = pauli.num_qubits
    qc = QuantumCircuit(n, n)
    pauli_str = pauli.to_label()
    l = len(pauli_str)
    for i in range(n) :
        if pauli_str[l-1-i] == 'X' :
            qc.h(i);
        if pauli_str[l-1-i] == 'Y' :
            qc.h(i); qc.s(i);
    return qc


def measure_rotation_for_pauli(pauli: Pauli):
    """ Apply rotations to convert arbitrary Pauli measurement into a Z-basis measurement. """
    n = pauli.num_qubits
    qc = QuantumCircuit(n, n)
    pauli_str = pauli.to_label()
    l = len(pauli_str)
    for i in range(n) :
        if pauli_str[l-1-i] == 'X' :
            qc.h(i);
        if pauli_str[l-1-i] == 'Y' :
            qc.sdg(i); qc.h(i);
    return qc


def expectation_from_counts(pauli: Pauli, counts: dict) :
    """ Compute expectation value of z-type Pauli string. """
    total_shots = 0
    n = pauli.num_qubits
    pauli_str = pauli.to_label()
    l = len(pauli_str)
    assert all(pauli.x == 0), "Expected Z-type Pauli string"
    expectation = 0.0
    for bitstring, count in counts.items() :
        term = 1
        for i in range(n) :
            if pauli_str[l-1-i] == 'Z' and bitstring[n-1-i] == '1' :
                term = -term
        expectation += term * count
        total_shots += count
    if pauli.phase == 2 : expectation = -expectation
    return expectation / total_shots


def qiskit_clifford_dfe(clifford: Clifford, data_qubits=None, num_samples: int=30) :
    """
    Sample `num_samples` state preparation and measurement subcircuits
    the direct fidelity estimation (DFE) of a given Clifford circuit.
    `data_qubits` is the set of qubits (in Qiskit convention) on
    which we randomize the Pauli string.
    """
    n = clifford.num_qubits
    result = []
    for sample_id in range(num_samples) :
        random_pauli_str = sample_random_pauli(n, data_qubits)
        pauli = Pauli( random_pauli_str )
        state_prep_layer = prep_eigenstate(pauli)
        pauli = pauli.evolve(clifford, frame='s')
        measurement_layer = measure_rotation_for_pauli(pauli)
        measurement_pauli = pauli.evolve(measurement_layer, frame='s')
        result.append((state_prep_layer, measurement_layer, measurement_pauli))
    return result
