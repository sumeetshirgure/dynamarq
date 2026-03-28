from dynamarq.benchmarks.gates.LongRangeCNOTSparse import LongRangeCNOTSparse
from dynamarq.benchmarks.gates.LongRangeCNOT import LongRangeCNOT
from dynamarq.benchmarks.gates.CNOTLadder import CNOTLadder
from dynamarq.benchmarks.gates.Fanout import Fanout
from qiskit_aer import AerSimulator


print("--- Testing Clifford DFE benchmarks ---")

print("---Testing guppy---")


from hugr.qsystem.result import QsysShot, QsysResult

from selene_sim import build, Quest, Stim


from selene_sim import DepolarizingErrorModel

error_model = DepolarizingErrorModel(
    random_seed=42,
    # single qubit gate error rate
    p_1q=1e-3,
    # two qubit gate error rate
    p_2q=1e-3,
    # set state preparation and measurement error rates to 0
    p_meas=5e-3,
    p_init=5e-3,
)

benchmark = Fanout(30)
circuits = benchmark.guppy_circuits() 
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)

    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        # error_model=error_model,
        n_qubits=61,
        n_shots=1000,
        n_processes=4,
    ))
    print(i)
    results.append(result)

print( benchmark.guppy_score(results) )

benchmark = CNOTLadder(30)
circuits = benchmark.guppy_circuits() 
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)

    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        # error_model=error_model,
        n_qubits=59,
        n_shots=1000,
        n_processes=4,
    ))
    print(i)
    results.append(result)

print( benchmark.guppy_score(results) )

print("---Testing qiskit---")

benchmark = CNOTLadder(2)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = CNOTLadder(5)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = CNOTLadder(30)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


benchmark = Fanout(2)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = Fanout(5)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = Fanout(30)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


benchmark = LongRangeCNOTSparse(2)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = LongRangeCNOTSparse(5)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = LongRangeCNOTSparse(30)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


benchmark = LongRangeCNOT(2)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = LongRangeCNOT(5)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

benchmark = LongRangeCNOT(30)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )

