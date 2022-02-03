import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import numpy as np
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize 
from noisyopt import minimizeSPSA


x, y = ds.load_boston(return_X_y=True)
x = MinMaxScaler().fit_transform(x)
y = (y - np.min(y)) / (np.max(y) - np.min(y))

def convert_data(data, qubits, test=False):
    cs = []
    for i in data:
        cir = cirq.Circuit()
        for j in range(len(qubits)):
            cir += cirq.rx(i[j] * np.pi).on(qubits[j])
            cir += cirq.ry(i[j] * np.pi).on(qubits[j])
        cs.append(cir)
    if test:
        return tfq.convert_to_tensor([cs])
    return tfq.convert_to_tensor(cs)

def encode(data, labels, qubits):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=43)
    return convert_data(X_train, qubits), convert_data(X_test, qubits), y_train, y_test

def layer(circuit, qubits, params):
    for i in range(len(qubits)):
        if i + 1 < len(qubits):
            circuit += cirq.CNOT(qubits[i], qubits[i + 1])
        circuit += cirq.ry(params[i * 2]).on(qubits[i])
        circuit += cirq.rz(params[i * 2 + 1]).on(qubits[i])
    circuit += cirq.CNOT(qubits[-1], qubits[0])
    return circuit

def model_circuit(qubits, depth):
    cir = cirq.Circuit()
    num_params = depth * 2 * len(qubits)
    params = sympy.symbols("q0:%d"%num_params)
    for i in range(depth):
        cir = layer(cir, qubits, params[i * 2 * len(qubits):i * 2 * len(qubits) + 2 * len(qubits)])
    #cir.with_noise(cirq.depolarize(p=0.01))
    return cir

def create_model(qs, d):
    c = model_circuit(qs, d)
    readout_operators = [cirq.Z(qs[0])]
    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = tfq.layers.PQC(c, readout_operators, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(inputs)
    vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
    return vqc

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

qs = [cirq.GridQubit(0, i) for i in range(13)]
X_train, X_test, y_train, y_test = encode(x, y, qs)
sizes = [2]
num_rep = 3
toler = 1e-4
depth = 4
max_iter = 1000

ops = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "SPSA"]
results = dict()
stds = dict()

for size in sizes:
    print(size, depth * 4)
    depth = size
    for method in ops:
        name = method + "_" + str(size)
        results[name] = []
        stds[name] = []
        print(method)
        error = 0
        errors = []
        for _ in range(num_rep):
            vqc = create_model(qs, depth)
            if method == "SPSA":
                ret = minimizeSPSA(f, x0=np.random.uniform(0, 2 * np.pi, vqc.trainable_variables[0].shape[0]), args=(vqc, X_train, y_train), niter=max_iter, paired=False)
            else:
                ret = minimize(f, x0=np.random.uniform(0, 2 * np.pi, vqc.trainable_variables[0].shape[0]), args=(vqc, X_train, y_train), \
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

for size in sizes:
    print(size, depth * 4)
    depth = size
    for i, opt in enumerate(grad_ops):
        for lr in learning_rates:
            name = names[i] + "_" + str(size) + "_" + str(lr)
            results[name] = []
            stds[name] = []
            opt.learning_rate.assign(lr)
            error = 0
            errors = []
            for _ in range(num_rep):
                vqc = create_model(qs, depth)
                final = train(vqc, X_train, toler, y_train, opt, max_iter)
                error += final
                errors.append(final)
            results[name].append(error/num_rep)
            stds[name].append(np.std(errors))
        for key, value in results.items():
            print(key, value, stds[key])
                
for key, value in results.items():
    print(key, value, stds[key])
