from dynamarq.benchmark import Benchmark

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit.quantum_info import hellinger_fidelity

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, panic
from guppylang.std.quantum import qubit, measure, measure_array, h, cx, cz, x, y, z


class FiveQubitCode(Benchmark) :
    """Represents the [[5, 1, 3]] five qubit code error correction benchmark.
    This benchmark evaluates how well the hardware preserves |0> and |1> states
    encoded in the five qubit XZZXI code.
    Performs one round of syndrome measurement and Z logical stabilizer measurement.

    We evaluate the Hellinger fidelity between the obtained distribution and the
    ideal distribution (depending on the initial state) as the score for this benchmark.
    """
    def __init__(self) :
        self.init_states = ['0', '1']

        # Taken from https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code#Encoding
        self.ket0 = {basis: 1.0/16 for basis in [
            "00000", "10010", "01001", "10100",
            "01010", "11011", "00110", "11000",
            "11101", "00011", "11110", "01111",
            "10001", "01100", "10111", "00101",
            ]}
        self.ket1 = {basis: 1.0/16 for basis in [
            "11111", "01101", "10110", "01011",
            "10101", "00100", "11001", "00111",
            "00010", "11100", "00001", "10000",
            "01110", "10011", "01000", "11010",
            ]}


    def name(self) :
        return f"FiveQubitCode"


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :

        circuits = []
        for init_state in self.init_states :
            data = QuantumRegister(5, 'data')
            anc  = QuantumRegister(4, 'anc')
            stab = QuantumRegister(1, 'stab')
            meas = ClassicalRegister(5, 'meas')
            syn  = ClassicalRegister(4, 'syn')
            op   = ClassicalRegister(1, 'op')

            circuit = QuantumCircuit(data, anc, stab, meas, syn, op)

            # Perform syndrome measurement.
            circuit.barrier()

            for i in range(4) :
                circuit.h(anc[i])

            circuit.cx(anc[0], data[0])
            circuit.cz(anc[0], data[1])
            circuit.cz(anc[0], data[2])
            circuit.cx(anc[0], data[3])

            circuit.cx(anc[1], data[1])
            circuit.cz(anc[1], data[2])
            circuit.cz(anc[1], data[3])
            circuit.cx(anc[1], data[4])

            circuit.cx(anc[2], data[2])
            circuit.cz(anc[2], data[3])
            circuit.cz(anc[2], data[4])
            circuit.cx(anc[2], data[0])

            circuit.cx(anc[3], data[3])
            circuit.cz(anc[3], data[4])
            circuit.cz(anc[3], data[0])
            circuit.cx(anc[3], data[1])

            for i in range(4) :
                circuit.h(anc[i])

            circuit.barrier()

            if mcm :
                for i in range(4) :
                    circuit.append(MidCircuitMeasure(), [anc[i]], [syn[i]])
            else :
                circuit.measure(anc, syn)

            if stretch_dd :
                for i in range(5) :
                    s = circuit.add_stretch(f"s_syn_{i}")
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])

            # NOTE : Inserting barrier before feedforward operations.
            circuit.barrier()

            # Perform decoding and apply error correction.

            with circuit.if_test((syn, 8)) :
                circuit.x(data[0])
            with circuit.if_test((syn, 1)) :
                circuit.x(data[1])
            with circuit.if_test((syn, 3)) :
                circuit.x(data[2])
            with circuit.if_test((syn, 6)) :
                circuit.x(data[3])
            with circuit.if_test((syn, 12)) :
                circuit.x(data[4])

            with circuit.if_test((syn, 13)) :
                circuit.y(data[0])
            with circuit.if_test((syn, 11)) :
                circuit.y(data[1])
            with circuit.if_test((syn, 7)) :
                circuit.y(data[2])
            with circuit.if_test((syn, 15)) :
                circuit.y(data[3])
            with circuit.if_test((syn, 14)) :
                circuit.y(data[4])

            with circuit.if_test((syn, 5)) :
                circuit.z(data[0])
            with circuit.if_test((syn, 10)) :
                circuit.z(data[1])
            with circuit.if_test((syn, 4)) :
                circuit.z(data[2])
            with circuit.if_test((syn, 9)) :
                circuit.z(data[3])
            with circuit.if_test((syn, 2)) :
                circuit.z(data[4])

            circuit.barrier()

            circuit.h(stab[0])

            for i in range(5) :
                circuit.cz(stab[0], data[i])

            circuit.h(stab[0])

            # NOTE : Inserting barrier before feedforward operations.
            circuit.barrier()

            if mcm :
                circuit.append(MidCircuitMeasure(), [stab[0]], [op[0]])
            else :
                circuit.measure(stab, op)

            if stretch_dd :
                for i in range(5) :
                    s = circuit.add_stretch(f"s_op_{i}")
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])

            circuit.barrier()

            if init_state == '0' :
                for i in range(5) :
                    with circuit.if_test((op, 1)) :
                        circuit.x(data[i])
            if init_state == '1' :
                for i in range(5) :
                    with circuit.if_test((op, 0)) :
                        circuit.x(data[i])

            circuit.barrier()

            circuit.measure(data, meas)

            circuits.append(circuit)

        return circuits


    def qiskit_score(self, counts_list) :
        ideal_dists = [self.ket0, self.ket1]

        fidelity_sum = 0.0
        for ideal_dist, counts in zip(ideal_dists, counts_list) :
            total_shots = sum(counts.values())

            device_hist = dict()
            for bitstr, count in counts.items() :
                data_qubits = bitstr[-1:-1-5:-1]
                if data_qubits not in device_hist :
                    device_hist[data_qubits] = 0
                device_hist[data_qubits] += count
            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}

            fidelity_sum += hellinger_fidelity(ideal_dist, device_dist)

        return fidelity_sum / len(counts_list)


    def guppy_circuits(self) :

        @guppy
        def five_qubit_code(data: array[qubit, 5] @owned,
                            anc: array[qubit, 4] @owned,
                            stab: qubit @owned,
                            init_state: bool, # False for |0>, True for |1>
                            ) -> None :

            for i in range(4) :
                h(anc[i])

            cx(anc[0], data[0])
            cz(anc[0], data[1])
            cz(anc[0], data[2])
            cx(anc[0], data[3])

            cx(anc[1], data[1])
            cz(anc[1], data[2])
            cz(anc[1], data[3])
            cx(anc[1], data[4])

            cx(anc[2], data[2])
            cz(anc[2], data[3])
            cz(anc[2], data[4])
            cx(anc[2], data[0])

            cx(anc[3], data[3])
            cz(anc[3], data[4])
            cz(anc[3], data[0])
            cx(anc[3], data[1])

            for i in range(4) :
                h(anc[i])

            syn = measure_array(anc)

            syndrome_value = 0
            index = 0
            for v in syn :
                if v :
                    syndrome_value += 2 ** index
                index += 1

            if syndrome_value == 8 :
                x(data[0])
            if syndrome_value == 1 :
                x(data[1])
            if syndrome_value == 3 :
                x(data[2])
            if syndrome_value == 6 :
                x(data[3])
            if syndrome_value == 12 :
                x(data[4])

            if syndrome_value == 13 :
                y(data[0])
            if syndrome_value == 11 :
                y(data[1])
            if syndrome_value == 7 :
                y(data[2])
            if syndrome_value == 15 :
                y(data[3])
            if syndrome_value == 14 :
                y(data[4])

            if syndrome_value == 5 :
                z(data[0])
            if syndrome_value == 10 :
                z(data[1])
            if syndrome_value == 4 :
                z(data[2])
            if syndrome_value == 9 :
                z(data[3])
            if syndrome_value == 2 :
                z(data[4])

            h(stab)

            for i in range(5) :
                cz(stab, data[i])

            h(stab)

            op = measure(stab)

            if op ^ init_state :
                for i in range(5) :
                    x(data[i])

            meas = measure_array(data)

            for v in meas : result('meas', v)

        @guppy
        def fqc0() -> None :
            data = array(qubit() for _ in range(5))
            anc  = array(qubit() for _ in range(4))
            stab = qubit()
            init_state = False
            five_qubit_code(data, anc, stab, init_state)

        @guppy
        def fqc1() -> None :
            data = array(qubit() for _ in range(5))
            anc  = array(qubit() for _ in range(4))
            stab = qubit()
            init_state = True
            five_qubit_code(data, anc, stab, init_state)

        return [fqc0, fqc1]




    def guppy_score(self, results_list) :
        ideal_dists = [self.ket0, self.ket1]

        fidelity_sum = 0.0

        for ideal_dist, results in zip(ideal_dists, results_list) :
            collated_counts = results.collated_counts()
            total_shots = sum(collated_counts.values())

            device_hist = dict()

            for key in collated_counts.keys() :
                string = key[0][1]
                freq = collated_counts[ (('meas', string),) ]
                if string not in device_hist :
                    device_hist[ string ] = 0
                device_hist[ string ] += freq

            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}

            fidelity_sum += hellinger_fidelity(ideal_dist, device_dist)

        return fidelity_sum / len(results_list)
