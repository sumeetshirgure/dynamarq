import random
from dynamarq.benchmark import Benchmark
from dynamarq.clifford_dfe import clifford_dfe, expectation_from_counts

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, comptime
from guppylang.std.quantum import qubit, measure_array, h, cx, s, sdg, x, z


class Fanout(Benchmark):
    """
    Represents the fanout benchmark parameterized by the number of target qubits n (> 1).

    Device performance is based on the direct fidelity estimate (DFE).
    Since this is a Clifford circuit, DFE is a scalable metric.
    """
    def __init__(self, num_qubits: int, num_dfe_samples: int=30) :
        assert num_qubits > 1, "Number of target qubits must be > 1."
        self.n = num_qubits
        self.num_dfe_samples = num_dfe_samples

        self.clifford_repr = self.reference_circuits()[0]

        data_qubits = [0] * (self.n) + [1] * (self.n+1) # Qiskit convention

        # Reproducible across instantiations
        random.seed(1)

        # Initialize DFE SPAM circuits.
        self.dfe_subcircuits = clifford_dfe(
                self.clifford_repr, data_qubits, num_dfe_samples)


    def name(self) :
        return f"Fanout_{self.n}"


    def reference_circuits(self) :
        rep_data = QuantumRegister(self.n+1, 'rep_data')
        rep_anc  = QuantumRegister(self.n, 'rep_anc')
        rep_c0   = ClassicalRegister(self.n+1, 'rep_c0')
        rep_c1   = ClassicalRegister(self.n, 'rep_c1')

        clifford_repr = QuantumCircuit(rep_data, rep_anc, rep_c0, rep_c1)
        for idx in range(1, self.n+1) :
            clifford_repr.cx(rep_data[0], rep_data[idx])

        return [clifford_repr]


    def dynamic_circuit(self, mcm=True, stretch_dd=False) :
        """
        Implements a dynamic circuit for a Fanout gate.
        https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.7.023120
        """
        data = QuantumRegister(self.n+1, 'data')
        anc  = QuantumRegister(self.n, 'anc')
        c0 = ClassicalRegister(self.n+1, 'c0')
        c1 = ClassicalRegister(self.n, 'c1')
        qc = QuantumCircuit(data, anc, c0, c1)

        for i in range(self.n) :
            if i % 2 == 0 :
                qc.h(anc[i])
        for i in range(self.n) :
            if i % 2 == 1 :
                qc.cx(data[i], anc[i])
        for i in range(self.n) :
            if i % 2 == 0 and i+1 <= self.n :
                qc.cx(anc[i], data[i+1])
        for i in range(self.n) :
            if i % 2 == 1 :
                qc.cx(data[i], anc[i])
        qc.cx(data[0], anc[0])
        for i in range(self.n) :
            if i % 2 == 1 and i+1 < self.n :
                qc.cx(data[i+1], anc[i+1])
        for i in range(self.n) :
            if i % 2 == 1 and i+1 <= self.n :
                qc.cx(anc[i], data[i+1])
        for i in range(self.n) :
            if i % 2 == 1 and i+1 < self.n :
                qc.cx(data[i+1], anc[i+1])
        for i in range(self.n) :
            if i % 2 == 1 :
                qc.h(anc[i])
        qc.barrier()

        if mcm :
            for i in range(self.n) :
                qc.append(MidCircuitMeasure(), [anc[i]], [c1[i]])
        else :
            qc.measure(anc, c1)

        if stretch_dd :
            for i in range(self.n+1) :
                s = qc.add_stretch(f"s_{i}")
                qc.delay(s, data[i])
                qc.x(data[i])
                qc.delay(s, data[i])
                qc.delay(s, data[i])
                qc.x(data[i])
                qc.delay(s, data[i])

        z_parity = expr.lift(c1[self.n-1-self.n%2])
        for i in range(self.n-1-self.n%2, 0, -2) :
            if i-2 >= 0 :
                z_parity = expr.bit_xor(c1[i-2], z_parity)
        with qc.if_test(z_parity) :
            qc.z(data[0])

        x_parity = expr.lift(c1[0])
        for i in range(0, self.n, 2) :
            with qc.if_test(x_parity) :
                if i+1 <= self.n :
                    qc.x(data[i+1])
                if i+2 <= self.n :
                    qc.x(data[i+2])
            if i + 2 < self.n :
                x_parity = expr.bit_xor(c1[i+2], x_parity)

        qc.barrier()
        return qc


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :
        """
        Get circuits to run on quantum hardware used in evaluating the benchmark score.
        """
        circuits = []
        dynamic_circuit = self.dynamic_circuit(mcm, stretch_dd)
        for sp_circ, meas_circ, meas_pauli, _, _ in self.dfe_subcircuits :
            qc = QuantumCircuit(self.n+self.n+1, self.n+self.n+1)
            qc.compose(sp_circ, range(self.n+self.n+1), inplace=True)
            qc.compose(dynamic_circuit, range(self.n+self.n+1), range(self.n+self.n+1), inplace=True)
            qc.compose(meas_circ, range(self.n+self.n+1), inplace=True)
            qc.measure(range(self.n+1), range(self.n+1))
            circuits.append(qc)
        return circuits


    def qiskit_score(self, counts_list) :
        """
        Compute the direct fidelity estimate (DFE) for the implemented Clifford circuit.
        """
        fidelity_sum = 0.0
        for dfe_circ, counts in zip(self.dfe_subcircuits, counts_list) :
            _, _, meas_pauli, _, _ = dfe_circ
            estimate = expectation_from_counts(meas_pauli, counts)
            fidelity_sum += estimate
        score = fidelity_sum / len(counts_list)
        return max(score, 0.0)


    def guppy_circuits(self) :

        @guppy.comptime
        def prep_circuit(
                data: array[qubit, comptime(self.n+1)]) -> None :
            pauli = self.dfe_subcircuits[dfe_index][3]
            pauli_string = pauli.to_label()
            l = len(pauli_string)
            for i in range(self.n+1) :
                if pauli_string[l-1-i] == 'X' :
                    h(data[i])
                if pauli_string[l-1-i] == 'Y' :
                    h(data[i])
                    s(data[i])

        @guppy.comptime
        def meas_circuit(
                data: array[qubit, comptime(self.n+1)]) -> None :
            pauli = self.dfe_subcircuits[dfe_index][4]
            pauli_string = pauli.to_label()
            l = len(pauli_string)
            for i in range(self.n+1) :
                if pauli_string[l-1-i] == 'X' :
                    h(data[i])
                if pauli_string[l-1-i] == 'Y' :
                    sdg(data[i])
                    h(data[i])

        circuits = []
        for dfe_index in range(self.num_dfe_samples) :
            @guppy
            def guppy_circuit() -> None :
                data = array(qubit() for _ in range(comptime(self.n+1)))
                anc = array(qubit() for _ in range(comptime(self.n)))
                prep_circuit(data)

                for i in range(comptime(self.n)) :
                    if i % 2 == 0 :
                        h(anc[i])
                for i in range(comptime(self.n)) :
                    if i % 2 == 1 :
                        cx(data[i], anc[i])
                for i in range(comptime(self.n)) :
                    if i % 2 == 0 and i+1 <= comptime(self.n) :
                        cx(anc[i], data[i+1])
                for i in range(comptime(self.n)) :
                    if i % 2 == 1 :
                        cx(data[i], anc[i])

                cx(data[0], anc[0])

                for i in range(comptime(self.n)) :
                    if i % 2 == 1 and i+1 < comptime(self.n) :
                        cx(data[i+1], anc[i+1])
                for i in range(comptime(self.n)) :
                    if i % 2 == 1 and i+1 <= comptime(self.n) :
                        cx(anc[i], data[i+1])
                for i in range(comptime(self.n)) :
                    if i % 2 == 1 and i+1 < comptime(self.n) :
                        cx(data[i+1], anc[i+1])
                for i in range(comptime(self.n)) :
                    if i % 2 == 1 :
                        h(anc[i])

                c = measure_array(anc)

                z_parity = c[comptime(self.n-1-self.n%2)]
                for i in range(comptime(self.n-1-self.n%2), 0, -2) :
                    if i-2 >= 0 :
                        z_parity = c[i-2] ^ z_parity
                if z_parity :
                    z(data[0])

                x_parity = c[0]
                for i in range(0, comptime(self.n), 2) :
                    if x_parity :
                        if i+1 <= comptime(self.n) :
                            x(data[i+1])
                        if i+2 <= comptime(self.n) :
                            x(data[i+2])
                    if i + 2 < comptime(self.n) :
                        x_parity = x_parity ^ c[i+2]

                meas_circuit(data)
                meas = measure_array(data)
                result('meas', meas)

            circuits.append( guppy_circuit.compile() )

        return circuits

        raise NotImplementedError("Direct fidelity estimation is not available for guppy")

    def guppy_score(self, results_list) :
        fidelity_sum = 0.0
        for dfe_subcircuit, results in zip(self.dfe_subcircuits, results_list) :
            collated_counts = results.collated_counts()
            total_shots = sum(collated_counts.values())
            device_hist = dict()
            for key in collated_counts.keys() :
                string = key[0][1]
                freq = collated_counts[ (('meas', string),) ]
                string += '0'*(self.n)
                string = string[::-1]
                if string not in device_hist :
                    device_hist[ string ] = 0
                device_hist[ string ] += freq
            fidelity_sum += expectation_from_counts(dfe_subcircuit[2], device_hist)
        return fidelity_sum / len(results_list)
