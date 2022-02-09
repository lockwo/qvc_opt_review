import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import random
from copy import deepcopy
from functools import reduce 
import operator
from scipy.optimize import minimize
from noisyopt import minimizeSPSA

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def layer(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit.append([cirq.rx(parameters[2*i]).on(qubits[i])])
        circuit.append([cirq.rz(parameters[2*i+1]).on(qubits[i])])
    for i in range(len(qubits)-1):
        circuit.append([cirq.CNOT(qubits[i], qubits[i+1])])
    return circuit

def ansatz(circuit, qubits, layers, parameters):
    for i in range(layers):
        p = parameters[2 * i * len(qubits):2 * (i + 1) * len(qubits)]
        circuit = layer(circuit, qubits, p)
    return circuit

def hamiltonian(circuit, qubits, ham):
    for i in range(len(qubits)):
        if ham[i] == "x":
            circuit.append(cirq.ry(-np.pi/2).on(qubits[i]))
        elif ham[i] == "y":
            circuit.append(cirq.rx(np.pi/2).on(qubits[i]))
    return circuit

def make_vqe(qubits, layers, parameters, ham):
    circuit = ansatz(cirq.Circuit(), qubits, layers, parameters)
    circuit = hamiltonian(circuit, qubits, ham)
    circuit = circuit.with_noise(cirq.depolarize(p=0.01))
    return circuit

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def expcost(qubits, ham):
    return prod([cirq.Z(qubits[i]) for i in range(len(qubits)) if ham[i] != "i"])

def real_min(h, w):
    for i in range(len(h)):
        for j in range(len(h[i])):
            if h[i][j] == "x":
                h[i][j] = np.array([[0, 1], [1, 0]])
            elif h[i][j] == "y":
                h[i][j] = np.array([[0, -1j], [1j, 0]])
            elif h[i][j] == "z":
                h[i][j] = np.array([[1, 0], [0, -1]])
            elif h[i][j] == "i":
                h[i][j] = np.array([[1, 0], [0, 1]])
    full = np.zeros(shape=(2**len(h[0]), 2**len(h[0]))).astype('complex128')
    for i in range(len(h)):
        op = np.kron(h[i][0], h[i][1])
        for j in range(2, len(h[i])):   
            op = np.kron(op, h[i][j])
        full += (w[i] * op)
    eig = np.real(np.linalg.eigvals(full))
    return sorted(eig)[0]

class VQE(tf.keras.layers.Layer):
    def __init__(self, num_weights, circuits, ops) -> None:
        super(VQE, self).__init__()
        self.w = tf.Variable(np.random.uniform(0, np.pi, (1, num_weights)), dtype=tf.float32)
        self.layers = [tfq.layers.ControlledPQC(circuits[i], ops[i], repetitions=1000, differentiator=tfq.differentiators.ParameterShift()) \
            for i in range(len(circuits))]

    def call(self, input):
        return sum([self.layers[i]([input, self.w]) for i in range(len(self.layers))])

def create_vqe(hamilton, h_weights, lay, numq):
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    cs = []
    op = []
    qubits = [cirq.GridQubit(0, i) for i in range(numq)]
    num_params = lay * 2 * numq
    params = sympy.symbols('vqe0:%d'%num_params)

    for i in range(len(hamilton)):
        readout_ops = h_weights[i] * expcost(qubits, hamilton[i])
        op.append(readout_ops)
        cs.append(make_vqe(qubits, lay, params, hamilton[i]))

    v = VQE(num_params, cs, op)(ins)
    vqe_model = tf.keras.models.Model(inputs=ins, outputs=v)
    return vqe_model

#@tf.function
def grad(inputs, opt, vqe):
    with tf.GradientTape() as tape:
        guess = vqe(inputs)
    grads = tape.gradient(guess, vqe.trainable_variables)
    opt.apply_gradients(zip(grads, vqe.trainable_variables))
    return guess

def train_vqe(vqe, tol, opt, inputs, max_iter):
    old = np.inf
    iterr = 0
    while True:
        guess = grad(inputs, opt, vqe)
        guess = guess.numpy()[0][0]
        if abs(old - guess) < tol:
            break
        if iterr > max_iter:
            break
        old = guess
        iterr += 1
    return guess

def create_hamil(decomp, qubits):
    possibilities = ["x", "y", "z"]
    hamilton = [[random.choice(possibilities) for _ in range(qubits)] for _ in range(decomp)]
    h_weights = [random.uniform(-1, 1) for _ in range(decomp)]
    return hamilton, h_weights

def f(x, *args):
    args[0].set_weights([np.expand_dims(x, axis=0)])
    return args[0](args[1]).numpy()[0][0]

decomp = 10
layers = 4 
num_rep = 2
toler = 1e-4
max_iter = 1000
sizes = [5, 10]
inputs = tfq.convert_to_tensor([cirq.Circuit()])

ops = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "SPSA"]

results = dict()
stds = dict()

for size in sizes:
    qu = size
    print(size, qu * layers * 2)
    for method in ops:
        name = method + "_" + str(size)
        results[name] = []
        stds[name] = []
        print(size, method)
        error = 0
        errors = []
        for n in range(num_rep):
            h, h_w = create_hamil(decomp, qu)
            vqe = create_vqe(h, h_w, layers, qu)
            if method == "SPSA":
                ret = minimizeSPSA(f, x0=np.random.uniform(0, 2 * np.pi, qu * layers * 2), args=(vqe, inputs), niter=max_iter, paired=False)
            else:
                ret = minimize(f, x0=np.random.uniform(0, 2 * np.pi, qu * layers * 2), args=(vqe, inputs), method=method, tol=toler, options={"maxiter" : max_iter})
            final = ret['fun']
            r = real_min(deepcopy(h), h_w)
            error += abs(r - final)
            errors.append(abs(r - final))
        results[name].append(error/num_rep)
        stds[name].append(np.std(errors))
        for key, value in results.items():
            print(key, value, stds[key])


for key, value in results.items():
    print(key, value, stds[key])

grad_ops = [tf.keras.optimizers.Adam(), tf.keras.optimizers.Adam(amsgrad=True), tf.keras.optimizers.Adadelta(), tf.keras.optimizers.Adagrad(), \
    tf.keras.optimizers.Adamax(), tf.keras.optimizers.Ftrl(), tf.keras.optimizers.Nadam(), tf.keras.optimizers.RMSprop(), tf.keras.optimizers.SGD()]
names = ["Adam", "AMSGrad", "Adadelta", "Adagrad", "Adamax", "Ftrl", "Nadam", "RMSProp", "SGD"]
xs = [-4, -3, -2, -1]
learning_rates = [10**i for i in xs]

results = dict()
stds = dict()

for size in sizes:
    qu = size
    print(size, qu * layers * 2)
    for i, opt in enumerate(grad_ops):
        for lr in learning_rates:
            name = names[i] + "_" + str(size) + "_" + str(lr)
            results[name] = []
            stds[name] = []
            print(i, opt, lr)
            opt.learning_rate.assign(lr)
            error = 0
            errors = []
            for _ in range(num_rep):
                h, h_w = create_hamil(decomp, qu)
                vqe = create_vqe(h, h_w, layers, qu)
                final = train_vqe(vqe, toler, opt, inputs, max_iter)
                r = real_min(deepcopy(h), h_w)
                error += abs(r - final)
                errors.append(abs(r - final))
            results[name].append(error/num_rep)
            stds[name].append(np.std(errors))
        for key, value in results.items():
            print(key, value, stds[key])

for key, value in results.items():
    print(key, value, stds[key])
