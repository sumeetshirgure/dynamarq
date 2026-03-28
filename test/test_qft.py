from dynamarq.benchmarks.algorithms.QFT import QFT
from dynamarq.benchmarks.algorithms.PartialQFT import PartialQFT

from qiskit_aer import AerSimulator

print("--- Testing QFT benchmarks ---")

benchmark = QFT(3)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = QFT(10)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )
# print(benchmark.qiskit_circuits(stretch_dd=True)[0])


benchmark = PartialQFT(3)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = PartialQFT(10)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )
# print(benchmark.qiskit_circuits(stretch_dd=True)[0])
