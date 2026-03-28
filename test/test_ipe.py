from dynamarq.benchmarks.algorithms.IPE import IPE
from qiskit_aer import AerSimulator

print("--- Testint IPE benchmarks ---")

benchmark = IPE(53, 7)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = IPE(int('1010101010', 2), 10)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


benchmark = IPE(int('1010101010', 2), 10)

circ = benchmark.qiskit_circuits(mcm=True, stretch_dd=True)
print(circ[0])
