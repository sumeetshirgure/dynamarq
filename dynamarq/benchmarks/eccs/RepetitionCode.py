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

    We evaluate the logical error rate on hardware as the score for this benchmark.
    Since the basic repetition code doesn't detect phase errors, it's a bit inaccurate
    when considering a generic error model like depolarizing noise.
    """
    def __init__(self, num_qubits: int) :
        self.n = num_qubits

        self.choices = [3, 5]
        self.init_states = ['1', '+']

        assert self.n in self.choices, f"Only n = {self.choices} are supported"

        self.corrections_3 = [ (1, (0,)), (2, (2,)), (3, (1,)) ]
        self.corrections_5 = [(1, (0,)), (3, (1,)), (6, (2,)), (12, (3,)),
                              (8, (4,)), (2, (0, 1)), (7, (0, 2)),
                              (13, (0, 3)), (9, (0, 4)), (5, (1, 2)),
                              (15, (1, 3)), (11, (1, 4)), (10, (2, 3)),
                              (14, (2, 4)), (4, (3, 4))]


    def name(self) :
        return f"RepetitionCode_{self.n}"


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

            circuit.barrier()

            # Measure syndrome.
            for i in range(self.n-1) :
                circuit.cx(data[i], anc[i])
                circuit.cx(data[i+1], anc[i])

            circuit.barrier()

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

            # NOTE : Inserting barrier before feed-forward just for ECC circuits.
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

            # Unprepare initial state.
            if init_state == '+' :
                # Unencode from repetition code.
                for i in range(self.n-1, 0, -1) :
                    circuit.cx(data[i-1], data[i])
                circuit.h(data[0])

            # Final readout.
            circuit.measure(data, meas)

            circuits.append(circuit)

        return circuits


    def qiskit_score(self, counts_list) :

        fidelity_sum = 0.0
        for init_state, counts in zip(self.init_states, counts_list) :
            total_shots = sum(counts.values())

            device_hist = dict()
            for bitstr, count in counts.items() :
                data_qubits = bitstr[-1:-1-self.n:-1]
                if data_qubits not in device_hist :
                    device_hist[data_qubits] = 0
                device_hist[data_qubits] += count
            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}

            if init_state == '1' :
                fidelity_sum += device_dist['1' * self.n]
            if init_state == '+' :
                fidelity_sum += device_dist['0' * self.n]

        return fidelity_sum / len(counts_list)


    def guppy_circuits(self) :

        @guppy
        def circuit_3_ket1() -> None :
            data = array(qubit() for _ in range(3))
            anc  = array(qubit() for _ in range(2))
            # Prepare initial state.
            x(data[0])
            # Encode into repetition code.
            cx(data[0], data[1])
            cx(data[1], data[2])
            # Measure syndrome.
            cx(data[0], anc[0])
            cx(data[1], anc[0])
            cx(data[1], anc[1])
            cx(data[2], anc[1])
            syndrome = measure_array(anc)
            # Apply correction.
            # corrections = ( (1, (0,)), (2, (2,)), (3, (1,)) )
            syndrome_value = 0
            index = 0
            for bit in syndrome :
                if bit : syndrome_value += 2**int(index)
                index += 1
            if syndrome_value == 1 : x(data[0])
            if syndrome_value == 2 : x(data[2])
            if syndrome_value == 3 : x(data[1])
            # Final readout
            meas = measure_array(data)
            for v in meas : result('meas', v)

        @guppy
        def circuit_3_plus() -> None :
            data = array(qubit() for _ in range(3))
            anc  = array(qubit() for _ in range(2))
            # Prepare initial state.
            h(data[0])
            # Encode into repetition code.
            cx(data[0], data[1])
            cx(data[1], data[2])
            # Measure syndrome.
            cx(data[0], anc[0])
            cx(data[1], anc[0])
            cx(data[1], anc[1])
            cx(data[2], anc[1])
            syndrome = measure_array(anc)
            # Apply correction.
            # corrections = ( (1, (0,)), (2, (2,)), (3, (1,)) )
            syndrome_value = 0
            index = 0
            for bit in syndrome :
                if bit : syndrome_value += 2**int(index)
                index += 1
            if syndrome_value == 1 : x(data[0])
            if syndrome_value == 2 : x(data[2])
            if syndrome_value == 3 : x(data[1])
            # Unencode from repetition code
            cx(data[1], data[2])
            cx(data[0], data[1])
            h(data[0])
            # Final readout
            meas = measure_array(data)
            for v in meas : result('meas', v)

        @guppy
        def circuit_5_ket1() -> None :
            data = array(qubit() for _ in range(5))
            anc  = array(qubit() for _ in range(4))
            # Prepare initial state.
            x(data[0])
            # Encode into repetition code.
            cx(data[0], data[1])
            cx(data[1], data[2])
            cx(data[2], data[3])
            cx(data[3], data[4])
            # Measure syndrome.
            cx(data[0], anc[0])
            cx(data[1], anc[0])
            cx(data[1], anc[1])
            cx(data[2], anc[1])
            cx(data[2], anc[2])
            cx(data[3], anc[2])
            cx(data[3], anc[3])
            cx(data[4], anc[3])
            syndrome = measure_array(anc)
            # Apply correction.
            # corrections =  [(1, (0,)), (3, (1,)), (6, (2,)), (12, (3,)),
            #                   (8, (4,)), (2, (0, 1)), (7, (0, 2)),
            #                   (13, (0, 3)), (9, (0, 4)), (5, (1, 2)),
            #                   (15, (1, 3)), (11, (1, 4)), (10, (2, 3)),
            #                   (14, (2, 4)), (4, (3, 4))]
            syndrome_value = 0
            index = 0
            for bit in syndrome :
                if bit : syndrome_value += 2**int(index)
                index += 1
            if syndrome_value == 1  : x(data[0])
            if syndrome_value == 3  : x(data[1])
            if syndrome_value == 6  : x(data[2])
            if syndrome_value == 12 : x(data[3])
            if syndrome_value == 8  : x(data[4])
            if syndrome_value == 2  : x(data[0]); x(data[1])
            if syndrome_value == 7  : x(data[0]); x(data[2])
            if syndrome_value == 13 : x(data[0]); x(data[3])
            if syndrome_value == 9  : x(data[0]); x(data[4])
            if syndrome_value == 5  : x(data[1]); x(data[2])
            if syndrome_value == 15 : x(data[1]); x(data[3])
            if syndrome_value == 11 : x(data[1]); x(data[4])
            if syndrome_value == 10 : x(data[2]); x(data[3])
            if syndrome_value == 14 : x(data[2]); x(data[4])
            if syndrome_value == 4  : x(data[3]); x(data[4])
            # Final readout
            meas = measure_array(data)
            for v in meas : result('meas', v)

        @guppy
        def circuit_5_plus() -> None :
            data = array(qubit() for _ in range(5))
            anc  = array(qubit() for _ in range(4))
            # Prepare initial state.
            h(data[0])
            # Encode into repetition code.
            cx(data[0], data[1])
            cx(data[1], data[2])
            cx(data[2], data[3])
            cx(data[3], data[4])
            # Measure syndrome.
            cx(data[0], anc[0])
            cx(data[1], anc[0])
            cx(data[1], anc[1])
            cx(data[2], anc[1])
            cx(data[2], anc[2])
            cx(data[3], anc[2])
            cx(data[3], anc[3])
            cx(data[4], anc[3])
            syndrome = measure_array(anc)
            # Apply correction.
            # corrections =  [(1, (0,)), (3, (1,)), (6, (2,)), (12, (3,)),
            #                   (8, (4,)), (2, (0, 1)), (7, (0, 2)),
            #                   (13, (0, 3)), (9, (0, 4)), (5, (1, 2)),
            #                   (15, (1, 3)), (11, (1, 4)), (10, (2, 3)),
            #                   (14, (2, 4)), (4, (3, 4))]
            syndrome_value = 0
            index = 0
            for bit in syndrome :
                if bit : syndrome_value += 2**int(index)
                index += 1
            if syndrome_value == 1  : x(data[0])
            if syndrome_value == 3  : x(data[1])
            if syndrome_value == 6  : x(data[2])
            if syndrome_value == 12 : x(data[3])
            if syndrome_value == 8  : x(data[4])
            if syndrome_value == 2  : x(data[0]); x(data[1])
            if syndrome_value == 7  : x(data[0]); x(data[2])
            if syndrome_value == 13 : x(data[0]); x(data[3])
            if syndrome_value == 9  : x(data[0]); x(data[4])
            if syndrome_value == 5  : x(data[1]); x(data[2])
            if syndrome_value == 15 : x(data[1]); x(data[3])
            if syndrome_value == 11 : x(data[1]); x(data[4])
            if syndrome_value == 10 : x(data[2]); x(data[3])
            if syndrome_value == 14 : x(data[2]); x(data[4])
            if syndrome_value == 4  : x(data[3]); x(data[4])
            # Final readout
            # Unencode from repetition code.
            cx(data[3], data[4])
            cx(data[2], data[3])
            cx(data[1], data[2])
            cx(data[0], data[1])
            h(data[0])
            meas = measure_array(data)
            for v in meas : result('meas', v)

        match self.n :
            case 3 : return [circuit_3_ket1, circuit_3_plus]
            case 5 : return [circuit_5_ket1, circuit_5_plus]

        raise ValueError(f"Only {self.choices} are supported.")


    def guppy_score(self, results_list) :

        fidelity_sum = 0.0
        for init_state, results in zip(self.init_states, results_list) :
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

            if init_state == '1' :
                fidelity_sum += device_dist['1' * self.n]
            if init_state == '+' :
                fidelity_sum += device_dist['0' * self.n]

        return fidelity_sum / len(results_list)
