from dynamarq.benchmark import Benchmark

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit.quantum_info import hellinger_fidelity

import guppylang
from guppylang import guppy


class GHZ(Benchmark) :
    """Represents the GHZ state preparation benchmark parameterized
    by the number of qubits n.

    Device performance is based on the Hellinger fidelity between
    the experimental and ideal probability distributions.
    """
    def __init__(self, num_qubits: int) :
        self.n = num_qubits
        assert self.n > 0, f"n(={num_qubits}) must be a positive integer"


    def reference_circuit(self) :
        data = QuantumRegister(self.n, 'data')
        meas = ClassicalRegister(self.n, 'meas')
        circuit = QuantumCircuit(data, meas)

        circuit.h(data[0])
        for i in range(self.n-1) :
            circuit.cx(i, i+1)

        circuit.measure(data, meas)
        return circuit


    def qiskit_circuit(self, mcm = True, stretch_dd = False) :
        data = QuantumRegister(self.n, 'data')
        anc  = QuantumRegister(self.n-1, 'anc')
        meas = ClassicalRegister(self.n, 'meas')
        cr   = ClassicalRegister(self.n-1, 'cr')

        circuit = QuantumCircuit(data, anc, meas, cr)

        for i in range(self.n) :
            circuit.h(data[i])

        for i in range(self.n-1) :
            circuit.cx(data[i], anc[i])

        for i in range(self.n-1) :
            circuit.cx(data[i+1], anc[i])

        circuit.barrier()

        if mcm :
            for i in range(self.n-1) :
                circuit.append(MidCircuitMeasure(), [anc[i]], [cr[i]])
        else :
            circuit.measure(anc, cr)

        if stretch_dd :
            for i in range(self.n) :
                s = circuit.add_stretch(f"s_{i}")
                circuit.delay(s, data[i])
                circuit.x(data[i])
                circuit.delay(s, data[i])
                circuit.delay(s, data[i])
                circuit.x(data[i])
                circuit.delay(s, data[i])

        circuit.barrier()

        for i in range(self.n-1) :
            if i == 0 :
                parity = expr.lift(cr[i])
            else :
                parity = expr.bit_xor(cr[i], parity)
            with circuit.if_test(parity) :
                circuit.x(data[i+1])

        circuit.barrier()

        circuit.measure(data, meas)

        return circuit


    def qiskit_score(self, counts: dict) -> float:
        """Compute the Hellinger fidelity between the experimental and ideal
        results, i.e., 50% probabilty of measuring the all-zero state and 50%
        probability of measuring the all-one state.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.n: 0.5 for b in ["0", "1"]}

        total_shots = sum(counts.values())

        device_hist = dict()
        for bitstr, count in counts.items() :
            data_qubits = bitstr[-1:-1-self.n:-1]

            if data_qubits not in device_hist :
                device_hist[data_qubits] = 0

            device_hist[data_qubits] += count

        device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}

        return hellinger_fidelity(ideal_dist, device_dist)
