import dynamarq
from dynamarq.benchmarks.gates.RepeatUntilSuccess import RepeatUntilSuccess
from qiskit_aer import AerSimulator

from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, IBMBackend
from qiskit_ibm_runtime.fake_provider import FakeFractionalBackend, FakeGuadalupeV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

print("---Testing qiskit AerSimulator---")

rus = RepeatUntilSuccess(1)
jobs = [AerSimulator().run(circuit, shots=4096)
        for circuit in rus.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print(rus.qiskit_score(counts))

rus = RepeatUntilSuccess(3)
jobs = [AerSimulator().run(circuit, shots=4096)
        for circuit in rus.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print(rus.qiskit_score(counts))

rus = RepeatUntilSuccess(5)
jobs = [AerSimulator().run(circuit, shots=4096)
        for circuit in rus.qiskit_circuits(mcm=False)]
counts = [job.result().get_counts() for job in jobs]
print(rus.qiskit_score(counts))

print("---Testing qiskit noisy simulator---")

rus = RepeatUntilSuccess(1)
circuit = rus.qiskit_circuits(mcm=False, stretch_dd=False)[0]
backend = FakeFractionalBackend()
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
isa_circuit = pm.run(circuit)

sampler = Sampler(backend)
result = sampler.run([isa_circuit], shots=4096).result()
print(rus.qiskit_score([result[0].data.c0.get_counts()]))
