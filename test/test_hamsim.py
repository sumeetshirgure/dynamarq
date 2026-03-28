from dynamarq.benchmarks.algorithms.TFIM import TFIM

from qiskit_aer import AerSimulator

print("--- Testing Hamiltonian simulation benchmarks ---")

print('--- Testing qiskit ---')

benchmark = TFIM(3, 2)
circuits = benchmark.qiskit_circuits(mcm=False)
jobs = [AerSimulator(method='matrix_product_state').run(circ, shots=10000) for circ in circuits]
counts = [job.result().get_counts() for job in jobs]
print(benchmark.qiskit_score(counts))

benchmark = TFIM(3, 5)
circuits = benchmark.qiskit_circuits(mcm=False)
jobs = [AerSimulator(method='matrix_product_state').run(circ, shots=10000) for circ in circuits]
counts = [job.result().get_counts() for job in jobs]
print(benchmark.qiskit_score(counts))

benchmark = TFIM(3, 20)
circuits = benchmark.qiskit_circuits(mcm=False)
jobs = [AerSimulator(method='matrix_product_state').run(circ, shots=10000) for circ in circuits]
counts = [job.result().get_counts() for job in jobs]
print(benchmark.qiskit_score(counts))


benchmark = TFIM(5, 2)
circuits = benchmark.qiskit_circuits(mcm=False)
jobs = [AerSimulator(method='matrix_product_state').run(circ, shots=10000) for circ in circuits]
counts = [job.result().get_counts() for job in jobs]
print(benchmark.qiskit_score(counts))

benchmark = TFIM(5, 5)
circuits = benchmark.qiskit_circuits(mcm=False)
jobs = [AerSimulator(method='matrix_product_state').run(circ, shots=10000) for circ in circuits]
counts = [job.result().get_counts() for job in jobs]
print(benchmark.qiskit_score(counts))

benchmark = TFIM(5, 20)
circuits = benchmark.qiskit_circuits(mcm=False)
jobs = [AerSimulator(method='matrix_product_state').run(circ, shots=10000) for circ in circuits]
counts = [job.result().get_counts() for job in jobs]
print(benchmark.qiskit_score(counts))


print('--- Testing guppy ---')

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

benchmark = TFIM(3, 2)
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Quest(), 
        error_model=error_model,
        n_qubits=5,
        n_shots=10000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )


benchmark = TFIM(5, 20)
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Quest(), 
        error_model=error_model,
        n_qubits=9,
        n_shots=10000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )



