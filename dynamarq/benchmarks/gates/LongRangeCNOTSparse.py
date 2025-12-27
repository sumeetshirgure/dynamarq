import random
from dynamarq.benchmark import Benchmark
from dynamarq.clifford_dfe import qiskit_clifford_dfe, expectation_from_counts

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure


class LongRangeCNOTSparse(Benchmark):
    """
    Represents the long range CNOT benchmark parameterized by the range
    of target qubit n (> 1).

    Device performance is based on the direct fidelity estimate (DFE).
    Since this is a Clifford circuit, DFE is a scalable metric.
    """
    def __init__(self, cnot_range: int, num_dfe_samples: int=30) -> None:
        assert cnot_range > 1, "Range of target qubit must be > 1."
        self.n = cnot_range

        self.clifford_repr = self.reference_circuits()[0]

        data_qubits = [0] * (self.n) + [1] * (self.n+1) # Qiskit convention

        # Reproducible across instantiations
        random.seed(1)

        # Initialize DFE SPAM circuits.
        self.qiskit_dfe_subcircuits = qiskit_clifford_dfe(
                self.clifford_repr, data_qubits, num_dfe_samples)


    def name(self) :
        return f"LongRangeCNOTSparse_{self.n}"


    def reference_circuits(self) :
        rep_data = QuantumRegister(self.n+1, 'rep_data')
        rep_anc  = QuantumRegister(self.n, 'rep_anc')
        rep_c0   = ClassicalRegister(self.n+1, 'rep_c0')
        rep_c1   = ClassicalRegister(self.n, 'rep_c1')

        clifford_repr = QuantumCircuit(rep_data, rep_anc, rep_c0, rep_c1)
        clifford_repr.cx(rep_data[0], rep_data[self.n])

        return [clifford_repr]


    def dynamic_circuit(self, mcm=True, stretch_dd=False) :
        """
        Implements a dynamic circuit for a long range CNOT gate.
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
            if i % 2 == 0 :
                if i + 1 < self.n :
                    qc.cx(data[i+1], anc[i+1])
        for i in range(self.n) :
            if i % 2 == 0 :
                if i + 1 < self.n :
                    qc.cx(anc[i], data[i+1])
        for i in range(self.n) :
            if i % 2 == 0 :
                if i + 1 < self.n :
                    qc.cx(data[i+1], anc[i+1])
        for i in range(self.n) :
            if i % 2 == 0 :
                if i + 1 < self.n :
                    qc.cx(anc[i], data[i+1])

        qc.cx(data[0], anc[0])
        qc.cx(anc[self.n-1], data[self.n])

        for i in range(self.n) :
            if i % 2 == 1 :
                if i + 1 < self.n :
                    qc.cx(anc[i], data[i+1])
        for i in range(self.n) :
            if i % 2 == 1 :
                if i + 1 < self.n :
                    qc.cx(data[i+1], anc[i+1])
        for i in range(self.n) :
            if i % 2 == 1 :
                if i + 1 < self.n :
                    qc.cx(anc[i], data[i+1])
        for i in range(self.n) :
            if i % 2 == 1 :
                if i + 1 < self.n :
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
            if i + 2 < self.n :
                x_parity = expr.bit_xor(c1[i+2], x_parity)
        with qc.if_test(x_parity) :
            qc.x(data[self.n])

        qc.barrier()
        return qc


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :
        """
        Get DFE circuits to run on quantum hardware used in evaluating the benchmark score.
        """
        circuits = []
        dynamic_circuit = self.dynamic_circuit(mcm, stretch_dd)
        for sp_circ, meas_circ, meas_pauli in self.qiskit_dfe_subcircuits :
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
        for dfe_circ, counts in zip(self.qiskit_dfe_subcircuits, counts_list) :
            sp_circ, meas_circ, meas_pauli = dfe_circ
            estimate = expectation_from_counts(meas_pauli, counts)
            fidelity_sum += estimate
        score = fidelity_sum / len(counts_list)
        return max(score, 0.0)


    def guppy_circuits(self) :
        raise NotImplementedError("Direct fidelity estimation is not available for guppy")


    def guppy_score(self, results) :
        raise NotImplementedError("Direct fidelity estimation is not available for guppy")
