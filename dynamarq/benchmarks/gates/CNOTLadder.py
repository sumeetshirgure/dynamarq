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

class CNOTLadder(Benchmark):
    """
    Represents the CNOT ladder benchmark parameterized by the number of qubits n (> 1).

    Device performance is based on the direct fidelity estimate (DFE).
    Since this is a Clifford circuit, DFE is a scalable metric.
    """
    def __init__(self, num_qubits: int, num_dfe_samples: int=30) :
        assert num_qubits > 1, "Number of qubits must be > 1."

        self.n = num_qubits
        self.num_dfe_samples = num_dfe_samples

        self.clifford_repr = self.reference_circuits()[0]

        data_qubits = [0] * (self.n-1) + [1] * self.n # Qiskit convention

        # Reproducible across instantiations
        random.seed(1)

        # Initialize DFE SPAM circuits.
        self.dfe_subcircuits = clifford_dfe(
                self.clifford_repr, data_qubits, num_dfe_samples)


    def name(self) :
        return f"CNOTLadder_{self.n}"


    def reference_circuits(self) :
        rep_data = QuantumRegister(self.n, 'rep_data')
        rep_anc  = QuantumRegister(self.n-1, 'rep_anc')
        rep_c0   = ClassicalRegister(self.n, 'rep_c0')
        rep_c1   = ClassicalRegister(self.n-1, 'rep_c1')
        
        clifford_repr = QuantumCircuit(
                rep_data, rep_anc, rep_c0, rep_c1)

        for idx in range(self.n-1, 0, -1) :
            clifford_repr.cx(rep_data[idx-1], rep_data[idx])

        return [clifford_repr]


    def dynamic_circuit(self, mcm=True, stretch_dd=False) :
        """
        Implements a dynamic circuit for a CNOT ladder.
        https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.7.023120
        """
        data = QuantumRegister(self.n, 'data')
        anc  = QuantumRegister(self.n-1, 'anc')
        c0   = ClassicalRegister(self.n, 'c0')
        c1   = ClassicalRegister(self.n-1, 'c1')
        qc   = QuantumCircuit(data, anc, c0, c1)

        for i in range(self.n-1) : qc.cx(data[i], anc[i])
        for i in range(self.n-1) : qc.cx(anc[i], data[i+1])
        for i in range(self.n-1) : qc.h(anc[i])
        qc.barrier()

        if mcm :
            for i in range(self.n-1) :
                qc.append(MidCircuitMeasure(), [anc[i]], [c1[i]])
        else :
            qc.measure(anc, c1)

        if stretch_dd :
            for i in range(self.n) :
                s = qc.add_stretch(f"s_{i}")
                qc.delay(s, data[i])
                qc.x(data[i])
                qc.delay(s, data[i])
                qc.delay(s, data[i])
                qc.x(data[i])
                qc.delay(s, data[i])

        parity = expr.lift(c1[self.n-2])
        for i in range(self.n-2, -1, -1) :
            with qc.if_test(parity) :
                qc.z(data[i])
            if i > 0 :
                parity = expr.bit_xor(c1[i-1], parity)
        qc.barrier()
        return qc


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :
        """
        Get qiskit circuits to run on quantum hardware used in evaluating the benchmark score.
        """
        circuits = []
        dynamic_circuit = self.dynamic_circuit(mcm, stretch_dd)
        for sp_circ, meas_circ, meas_pauli, _, _ in self.dfe_subcircuits :
            qc = QuantumCircuit(self.n+self.n-1, self.n+self.n-1)
            qc.compose(sp_circ, range(self.n+self.n-1), inplace=True)
            qc.compose(dynamic_circuit, range(self.n+self.n-1), range(self.n+self.n-1), inplace=True)
            qc.compose(meas_circ, range(self.n+self.n-1), inplace=True)
            qc.measure(range(self.n), range(self.n))
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
                data: array[qubit, comptime(self.n)]) -> None :
            pauli = self.dfe_subcircuits[dfe_index][3]
            pauli_string = pauli.to_label()
            l = len(pauli_string)
            for i in range(self.n) :
                if pauli_string[l-1-i] == 'X' :
                    h(data[i])
                if pauli_string[l-1-i] == 'Y' :
                    h(data[i])
                    s(data[i])

        @guppy.comptime
        def meas_circuit(
                data: array[qubit, comptime(self.n)]) -> None :
            pauli = self.dfe_subcircuits[dfe_index][4]
            pauli_string = pauli.to_label()
            l = len(pauli_string)
            for i in range(self.n) :
                if pauli_string[l-1-i] == 'X' :
                    h(data[i])
                if pauli_string[l-1-i] == 'Y' :
                    sdg(data[i])
                    h(data[i])

        circuits = []
        for dfe_index in range(self.num_dfe_samples) :
            @guppy
            def guppy_circuit() -> None :
                data = array(qubit() for _ in range(comptime(self.n)))
                anc = array(qubit() for _ in range(comptime(self.n-1)))
                prep_circuit(data)

                for i in range(comptime(self.n)-1) :
                    cx(data[i], anc[i])
                for i in range(comptime(self.n)-1) :
                    cx(anc[i], data[i+1])
                for i in range(comptime(self.n)-1) :
                    h(anc[i])

                c = measure_array(anc)

                parity = c[comptime(self.n)-2]
                for i in range(comptime(self.n)-2, -1, -1) :
                    if parity :
                        z(data[i])
                    if i > 0 :
                        parity = parity ^ c[i-1]
 
                meas_circuit(data)
                meas = measure_array(data)
                result('meas', meas)

            circuits.append( guppy_circuit.compile() )

        return circuits


    def guppy_score(self, results_list) :
        fidelity_sum = 0.0
        for dfe_subcircuit, results in zip(self.dfe_subcircuits, results_list) :
            collated_counts = results.collated_counts()
            total_shots = sum(collated_counts.values())
            device_hist = dict()
            for key in collated_counts.keys() :
                string = key[0][1]
                freq = collated_counts[ (('meas', string),) ]
                string += '0'*(self.n-1)
                string = string[::-1]
                if string not in device_hist :
                    device_hist[ string ] = 0
                device_hist[ string ] += freq
            fidelity_sum += expectation_from_counts(dfe_subcircuit[2], device_hist)
        return fidelity_sum / len(results_list)
