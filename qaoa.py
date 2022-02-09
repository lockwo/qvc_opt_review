import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from noisyopt import minimizeSPSA


def mixing_hamiltonian(qubits, par):
    c = cirq.Circuit()
    for i in range(len(qubits)):
        c += cirq.rx(2 * par).on(qubits[i])
    return c

def cost_hamiltonian(qubits, g, ps):
    c = cirq.Circuit()
    for edge in g.edges():
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
        c += cirq.rz(ps).on(qubits[edge[1]])
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
    return c

def make_circuit(nodes, p, g):
    qs = [cirq.GridQubit(0, i) for i in range(nodes)]
    qaoa_circuit = cirq.Circuit()
    for i in qs:
        qaoa_circuit += cirq.H(i)
    num_param = 2 * p 
    qaoa_parameters = sympy.symbols("q0:%d"%num_param)
    for i in range(p):
        qaoa_circuit += cost_hamiltonian(qs, g, qaoa_parameters[2 * i])
        qaoa_circuit += mixing_hamiltonian(qs, qaoa_parameters[2 * i + 1])
    qaoa = qaoa_circuit.with_noise(cirq.depolarize(p=0.01))
    return qaoa_circuit

def cc(qubits, g):
    c = 0
    for edge in g.edges():
        c += cirq.PauliString(1/2 * cirq.Z(qubits[edge[0]]) * cirq.Z(qubits[edge[1]]))
    return c

def create_qaoa(g, p, nodes): 
    qs = [cirq.GridQubit(0, i) for i in range(nodes)]
    cost = cc(qs, g)
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    outs = tfq.layers.PQC(make_circuit(nodes, p, g), cost, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(ins)
    #layer1 = tfq.layers.NoisyPQC(model_circuit(nodes, p, g), cost, repetitions=1000, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(inputs)
    qaoa = tf.keras.models.Model(inputs=ins, outputs=outs)
    return qaoa

def train_qaoa(qaoa, inputs, tolerance, max_iter):
    old = 100
    iterr = 0
    while True:
        with tf.GradientTape() as tape:
            error = qaoa(inputs)
        grads = tape.gradient(error, qaoa.trainable_variables)
        opt.apply_gradients(zip(grads, qaoa.trainable_variables))
        error = error.numpy()[0][0]
        if abs(old - error) < tolerance:
            break
        if iterr > max_iter:
            break
        old = error
        iterr += 1
    return error

def f(x, *args):
    args[0].set_weights([x])
    return args[0](args[1]).numpy()[0][0]

inputs = tfq.convert_to_tensor([cirq.Circuit()])
max_iter = 1000
regularity = 4
graphs = [nx.random_regular_graph(n=5, d=regularity), nx.random_regular_graph(n=10, d=regularity), nx.random_regular_graph(n=15, d=regularity)]
num_rep = 3
toler = 1e-4
depth = 4

ops = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "SPSA"]
results = dict()
stds = dict()

for size, graph in enumerate(graphs):
    for method in ops:
        name = method + str(size)
        results[name] = []
        stds[name] = []
        print(size, method)
        error = 0
        errors = []
        for _ in range(num_rep):
            qaoa = create_qaoa(graph, depth, len(graph.nodes()))
            if method == "SPSA":
                ret = minimizeSPSA(f, x0=np.random.uniform(0, 2 * np.pi, depth * 2), args=(qaoa, inputs), niter=max_iter, paired=False)
            else:
                ret = minimize(f, x0=np.random.uniform(0, 2 * np.pi, depth * 2), args=(qaoa, inputs), method=method, tol=toler, options={"maxiter" : max_iter})
            final = ret['fun']
            error += final
            errors.append(final)
        results[name].append(error/num_rep)
        stds[name].append(np.std(errors))
        for key, value in results.items():
            print(key, value, stds[key])

for key, value in results.items():
    print(key, value,stds[key])

grad_ops = [tf.keras.optimizers.Adam(), tf.keras.optimizers.Adam(amsgrad=True), tf.keras.optimizers.Adadelta(), tf.keras.optimizers.Adagrad(), \
    tf.keras.optimizers.Adamax(), tf.keras.optimizers.Ftrl(), tf.keras.optimizers.Nadam(), tf.keras.optimizers.RMSprop(), tf.keras.optimizers.SGD()]
names = ["Adam", "AMSGrad", "Adadelta", "Adagrad", "Adamax", "Ftrl", "Nadam", "RMSProp", "SGD"]
xs = [-4, -3, -2, -1]
learning_rates = [10**i for i in xs]

results = dict()
stds = dict()

for size, graph in enumerate(graphs):
    print(size, depth * 2)
    for i, opt in enumerate(grad_ops):
        for lr in learning_rates:
            name = names[i] + "_" + str(size) + "_" + str(lr)
            results[name] = []
            stds[name] = []
            opt.learning_rate.assign(lr)
            error = 0
            errors = []
            for _ in range(num_rep):
                qaoa = create_qaoa(graph, depth, len(graph.nodes()))
                print(qaoa)
                final = train_qaoa(qaoa, inputs, toler, max_iter)
                error += final
                errors.append(final)
            results[name].append(error/num_rep)
            stds[name].append(np.std(errors))
        for key, value in results.items():
            print(key, value, stds[key])

for key, value in results.items():
    print(key, value, stds[key])
