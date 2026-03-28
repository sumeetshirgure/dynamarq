from dynamarq.benchmarks.eccs.RepetitionCode import RepetitionCode
from dynamarq.benchmarks.eccs.FiveQubitCode import FiveQubitCode
from dynamarq.benchmarks.eccs.SteaneCode import SteaneCode

from qiskit_aer import AerSimulator

print("--- Testing error correcting code benchmarks ---")

print('--- Testing qiskit ---')

benchmark = RepetitionCode(3)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


benchmark = RepetitionCode(5)
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


benchmark = FiveQubitCode()
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


benchmark = SteaneCode()
jobs = [AerSimulator().run(circuit, shots=1000) for circuit in benchmark.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print( benchmark.qiskit_score(counts) )


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

benchmark = RepetitionCode(3)
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
    results.append(result)
print( benchmark.guppy_score(results) )

benchmark = RepetitionCode(5)
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
    results.append(result)
print( benchmark.guppy_score(results) )

benchmark = RepetitionCode(3)
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        error_model=error_model,
        n_qubits=61,
        n_shots=1000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )

benchmark = RepetitionCode(5)
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        error_model=error_model,
        n_qubits=61,
        n_shots=1000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )



benchmark = FiveQubitCode()
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        # error_model=error_model,
        n_qubits=11,
        n_shots=16000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )

benchmark = FiveQubitCode()
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        error_model=error_model,
        n_qubits=11,
        n_shots=16000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )

benchmark = SteaneCode()
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        # error_model=error_model,
        n_qubits=14,
        n_shots=16000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )

benchmark = SteaneCode()
circuits = benchmark.guppy_circuits()
results = []
for i, circ in enumerate(circuits) :
    runner = build(circ)
    result = QsysResult(runner.run_shots(
        simulator=Stim(), 
        error_model=error_model,
        n_qubits=14,
        n_shots=16000,
        n_processes=4,
    ))
    results.append(result)
print( benchmark.guppy_score(results) )
