import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
from scipy.optimize import minimize
from noisyopt import minimizeSPSA


# Code from https://www.tensorflow.org/quantum/tutorials/qcnn
def generate_data(qubits):
    n_rounds = 20  # Produces n_rounds * n_qubits datapoints.
    excitations = []
    labels = []
    for n in range(n_rounds):
        for bit in qubits:
            rng = np.random.uniform(-np.pi, np.pi)
            excitations.append(cirq.Circuit(cirq.rx(rng)(bit)))
            labels.append(1 if (-np.pi / 2) <= rng <= (np.pi / 2) else -1)

    split_ind = int(len(excitations) * 0.7)
    train_excitations = excitations[:split_ind]
    test_excitations = excitations[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    return tfq.convert_to_tensor(train_excitations), np.array(train_labels), \
        tfq.convert_to_tensor(test_excitations), np.array(test_labels)

def cluster_state_circuit(bits):
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(bits))
    for this_bit, next_bit in zip(bits, bits[1:] + [bits[0]]):
        circuit.append(cirq.CZ(this_bit, next_bit))
    return circuit

def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])

def two_qubit_unitary(bits, symbols):
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    circuit += [cirq.ZZ(*bits)**symbols[6]]
    circuit += [cirq.YY(*bits)**symbols[7]]
    circuit += [cirq.XX(*bits)**symbols[8]]
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])
    return circuit

def two_qubit_pool(source_qubit, sink_qubit, symbols):
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit

def quantum_conv_circuit(bits, symbols):
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], symbols)
    return circuit

def quantum_pool_circuit(source_bits, sink_bits, symbols):
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, symbols)
    return circuit

def create_model_circuit(qubits):
    model_circuit = cirq.Circuit()
    symbols = sympy.symbols('qconv0:63')
    model_circuit += quantum_conv_circuit(qubits, symbols[0:15])
    model_circuit += quantum_pool_circuit(qubits[:4], qubits[4:], symbols[15:21])
    model_circuit += quantum_conv_circuit(qubits[4:], symbols[21:36])
    model_circuit += quantum_pool_circuit(qubits[4:6], qubits[6:], symbols[36:42])
    model_circuit += quantum_conv_circuit(qubits[6:], symbols[42:57])
    model_circuit += quantum_pool_circuit([qubits[6]], [qubits[7]], symbols[57:63])
    model_circuit = model_circuit.with_noise(cirq.depolarize(p=0.01))
    return model_circuit

def make_model():
    cluster_state_bits = cirq.GridQubit.rect(1, 8)
    readout_operators = cirq.Z(cluster_state_bits[-1])
    excitation_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    cluster_state = tfq.layers.AddCircuit()(excitation_input, prepend=cluster_state_circuit(cluster_state_bits))

    quantum_model = tfq.layers.PQC(create_model_circuit(cluster_state_bits),readout_operators)(cluster_state)

    qcnn_model = tf.keras.Model(inputs=[excitation_input], outputs=[quantum_model])
    return qcnn_model

def train(vqc, inputs, tolerance, labels, opt, max_iter):
    old = 100
    iterr = 0
    while True:
        with tf.GradientTape() as tape:
            guess = vqc(inputs)
            error = tf.math.reduce_mean(tf.math.square(guess - labels))
        grads = tape.gradient(error, vqc.trainable_variables)
        opt.apply_gradients(zip(grads, vqc.trainable_variables))
        error = error.numpy()
        if abs(old - error) < tolerance:
            break
        if iterr > max_iter:
            break
        iterr += 1
        old = error
    return error

def f(x, *args):
    args[0].set_weights([x])
    return tf.math.reduce_mean(tf.math.square(args[0](args[1]) - args[2])).numpy()

num_rep = 3
toler = 1e-4
max_iter = 1000
train_excitations, train_labels, test_excitations, test_labels = generate_data(cirq.GridQubit.rect(1, 8))

ops = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "SPSA"]
results = dict()
stds = dict()

for method in ops:
    name = method
    results[name] = []
    stds[name] = []
    print(method)
    error = 0
    errors = []
    for _ in range(num_rep):
        vqc = make_model()
        if method == "SPSA":
            ret = minimizeSPSA(f, x0=np.random.uniform(0, 2 * np.pi, vqc.trainable_variables[0].shape[0]), args=(vqc, train_excitations, train_labels), niter=max_iter, paired=False)
        else:
            ret = minimize(f, x0=np.random.uniform(0, 2 * np.pi, vqc.trainable_variables[0].shape[0]), args=(vqc, train_excitations, train_labels), \
                method=method, tol=toler, options={"maxiter" : max_iter})
        final = ret['fun']
        error += final
        errors.append(final)
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

for i, opt in enumerate(grad_ops):
    for lr in learning_rates:
        name = names[i] +  "_" + str(lr)
        results[name] = []
        stds[name] = []
        opt.learning_rate.assign(lr)
        error = 0
        errors = []
        for _ in range(num_rep):
            vqc = make_model()
            final = train(vqc, train_excitations, toler, train_labels, opt, max_iter)
            error += final
            errors.append(final)
        results[name].append(error/num_rep)
        stds[name].append(np.std(errors))
    for key, value in results.items():
        print(key, value, stds[key])


for key, value in results.items():
    print(key, value, stds[key])
