from dynamarq.benchmark import Benchmark

from math import pi

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit_aer import AerSimulator


class TFIM(Benchmark) :
    """Represents a quantum simulation benchmark for the 1D transverse
    field Ising model parameterized by the number of sites and the number
    of Trotter steps.
    
    Device performance is based on the relative error in the hardware
    computed average magnetization versus that from ideal circuit simulation
    based on matrix product states.

    The TFIM Hamiltonian reads: H = -J \\sum_i Z_i Z_{i+1} + h \\sum_i X_i
    It is Trotterized to the second order as e^(a+b) ~ e^(b/2) e^a e^(b/2)
    """
    def __init__(self, num_sites: int, num_steps: int) :
        self.n = num_sites
        self.steps = num_steps

        self.default_h = -7.0
        self.default_J = 1.0
        self.default_dt = 2 * pi * 1 / 30 * 0.25
    

    def reference_circuits(self, h=None, J=None, dt=None) :
        if h is None : h = self.default_h
        if J is None : J = self.default_J
        if dt is None : dt = self.default_dt

        theta_x = h * dt
        theta_zz = -2 * J * dt

        data = QuantumRegister(self.n)
        meas = ClassicalRegister(self.n)
        circuit = QuantumCircuit(data, meas)

        for step in range(self.steps):
            for i in range(self.n):
                circuit.rx(theta_x, data[i])
            for i in range(0, self.n-1, 2) :
                circuit.rzz(theta_zz, data[i], data[i+1])
            for i in range(1, self.n-1, 2) :
                circuit.rzz(theta_zz, data[i], data[i+1])
            for i in range(self.n):
                circuit.rx(theta_x, data[i])

        circuit.measure(data, meas)

        return [circuit]


    def qiskit_circuits(self, h=None, J=None, dt=None, mcm=True, stretch_dd=False) :
        if h is None : h = self.default_h
        if J is None : J = self.default_J
        if dt is None : dt = self.default_dt

        theta_x = h * dt
        theta_zz = -2 * J * dt

        data = QuantumRegister(self.n, 'data')
        anc  = QuantumRegister(self.n-1, 'anc')
        meas = ClassicalRegister(self.n, 'meas')
        c1   = ClassicalRegister(self.n-1, 'c1')

        circuit = QuantumCircuit(data, anc, meas, c1)

        for step in range(self.steps) :

            for i in range(self.n) :
                circuit.rx(theta_x, data[i])
            for i in range(0, self.n-1, 2) :
                circuit.cx(data[i], anc[i])
                circuit.cx(data[i+1], anc[i])
                circuit.rz(theta_zz, anc[i])
                circuit.h(anc[i])
            for i in range(1, self.n-1, 2) :
                circuit.cx(data[i], anc[i])
                circuit.cx(data[i+1], anc[i])
                circuit.rz(theta_zz, anc[i])
                circuit.h(anc[i])

            circuit.barrier()

            for i in range(self.n-1) :
                if mcm :
                    circuit.append(MidCircuitMeasure(), [anc[i]], [c1[i]])
                else :
                    circuit.measure(anc[i], c1[i])

            if stretch_dd :
                for i in range(self.n) :
                    s = circuit.add_stretch(f"s_{step}_{i}")
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])

            for i in range(0, self.n-1, 2) :
                flag = expr.lift(c1[i])
                with circuit.if_test(flag) :
                    circuit.z(data[i])
                    circuit.z(data[i+1])

            for i in range(1, self.n-1, 2) :
                flag = expr.lift(c1[i])
                with circuit.if_test(flag) :
                    circuit.z(data[i])
                    circuit.z(data[i+1])

            circuit.barrier()

            for i in range(self.n) :
                circuit.rx(theta_x, data[i])

        circuit.measure(data, meas)

        return [circuit]


    def average_magnetization(self, counts):
        mag = 0.0
        for index in range(self.n) :
            z_exp = 0.0
            tot = 0.0
            for bitstring, value in counts.items():
                bit = int(bitstring[-1-index])
                sign = 1 if bit == 0 else -1
                z_exp += sign * value
                tot += value
            z_exp /= tot
            mag += z_exp
        return mag / self.n


    def qiskit_score(self, counts_list) :
        mz_ideal = self.average_magnetization(AerSimulator(method='matrix_product_state').run(
                self.reference_circuits()[0], shots=10000).result().get_counts())
        mz_exp = self.average_magnetization(counts_list[0])
        return 1 - abs(mz_ideal - mz_exp) / 2
