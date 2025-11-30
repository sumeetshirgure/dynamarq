from dynamarq.benchmark import Benchmark

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit.quantum_info import hellinger_fidelity

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, panic
from guppylang.std.quantum import qubit, measure, measure_array, h, cx, x


class RepetitionCode(Benchmark) :
    """Represents a repetition code error correction benchmark parameterized
    by the number of redundant qubits `n`.
    This benchmark evaluates how well the hardware preserves |1> and |+> states
    and performs one round of syndrome measurement and correction.

    We evaluate the Hellinger fidelity between the obtained distribution and the
    ideal distribution (depending on the initial state) as the score for this benchmark.
    Since the basic repetition code doesn't detect phase errors, it's a bit inaccurate
    when considering a generic error model like depolarizing noise.
    """
    def __init__(self, num_qubits: int) :
        self.n = num_qubits

        self.choices = [3, 5]
        self.init_states = ['1', '+']

        assert self.n in self.choices, f"Only {self.choices} are supported"

        self.corrections_3 = [ (1, (0,)), (2, (2,)), (3, (1,)) ]
        self.corrections_5 = [(1, (0,)), (3, (1,)), (6, (2,)), (12, (3,)),
                              (8, (4,)), (2, (0, 1)), (7, (0, 2)),
                              (13, (0, 3)), (9, (0, 4)), (5, (1, 2)),
                              (15, (1, 3)), (11, (1, 4)), (10, (2, 3)),
                              (14, (2, 4)), (4, (3, 4))]


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :

        circuits = []
        for init_state in self.init_states :
            data = QuantumRegister(self.n, 'data')
            anc = QuantumRegister(self.n-1, 'anc')
            meas = ClassicalRegister(self.n, 'meas')
            syn = ClassicalRegister(self.n-1, 'syn')
            circuit = QuantumCircuit(data, anc, meas, syn)

            # Prepare initial state.
            if init_state == '1' : circuit.x(data[0])
            if init_state == '+' : circuit.h(data[0])

            # Encode into repetition code.
            for i in range(1, self.n) :
                circuit.cx(data[i-1], data[i])

            circuit.barrier(data)

            # Measure syndrome.
            for i in range(self.n-1) :
                circuit.cx(data[i], anc[i])
                circuit.cx(data[i+1], anc[i])

            if mcm :
                for i in range(self.n-1) :
                    circuit.append(MidCircuitMeasure(), [anc[i]], [syn[i]])
            else :
                circuit.measure(anc, syn)

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

            # Decode error.
            match self.n :
                case 3 : corrections = self.corrections_3
                case 5 : corrections = self.corrections_5

            # Apply correction.
            for syndrome_value, correction_vector in corrections :
                with circuit.if_test((syn, syndrome_value)) :
                    for qubit_index in correction_vector :
                        circuit.x(data[qubit_index])
            circuit.barrier()

            # Final readout.
            circuit.measure(data, meas)

            circuits.append(circuit)

        return circuits


    def qiskit_score(self, counts_list) :
        ket1 = {b * self.n: 1 for b in ["1"]}
        plus = {b * self.n: 0.5 for b in ["0", "1"]}

        ideal_dists = [ket1, plus]

        fidelity_sum = 0.0
        for ideal_dist, counts in zip(ideal_dists, counts_list) :
            total_shots = sum(counts.values())

            device_hist = dict()
            for bitstr, count in counts.items() :
                data_qubits = bitstr[-1:-1-self.n:-1]
                if data_qubits not in device_hist :
                    device_hist[data_qubits] = 0
                device_hist[data_qubits] += count
            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}
            
            fidelity_sum += hellinger_fidelity(ideal_dist, device_dist)

        return fidelity_sum / len(counts_list)
