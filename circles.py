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


circle_data, circle_labels = ds.make_circles(300, noise=0.2, factor=0.3, shuffle=True)
circle_data = MinMaxScaler().fit_transform(circle_data)

def convert_data(data, qubits, test=False):
    cs = []
    for i in data:
        cir = cirq.Circuit()
        cir += cirq.rx(i[0] * np.pi).on(qubits[0])
        cir += cirq.rz(i[0] * np.pi).on(qubits[0])
        cir += cirq.rx(i[1] * np.pi).on(qubits[1])
        cir += cirq.rz(i[1] * np.pi).on(qubits[1])
        cs.append(cir)
    if test:
        return tfq.convert_to_tensor([cs])
    return tfq.convert_to_tensor(cs)

def encode(data, labels, qubits):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=43)
    return convert_data(X_train, qubits), convert_data(X_test, qubits), y_train, y_test

def layer(circuit, qubits, params):
    circuit += cirq.CNOT(qubits[0], qubits[1])
    circuit += cirq.ry(params[0]).on(qubits[0])
    circuit += cirq.ry(params[1]).on(qubits[1])
    circuit += cirq.rz(params[2]).on(qubits[0])
    circuit += cirq.rz(params[3]).on(qubits[1])
    return circuit

def model_circuit(qubits, depth):
    cir = cirq.Circuit()
    num_params = depth * 4
    params = sympy.symbols("q0:%d"%num_params)
    for i in range(depth):
        cir = layer(cir, qubits, params[i * 4:i * 4 + 4])
    cir = cir.with_noise(cirq.depolarize(p=0.01))
    return cir

def create_model(qs, d):
    readout_operators = [cirq.Z(qs[0])]
    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = tfq.layers.PQC(model_circuit(qs, d), readout_operators, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(inputs)
    vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
    return vqc

def train(vqc, inputs, tolerance, labels, opt, maxiter):
    old = 100
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    iterr = 0   
    while True:
        with tf.GradientTape() as tape:
            guess = vqc(inputs)
            error = loss_fn(tf.squeeze(labels), guess)
        grads = tape.gradient(error, vqc.trainable_variables)
        opt.apply_gradients(zip(grads, vqc.trainable_variables))
        error = error.numpy()
        if abs(old - error) < tolerance:
            break
        if iterr > maxiter:
            break
        iterr += 1
        old = error
    return error

def f(x, *args):
    args[0].set_weights([x])
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(args[2], tf.squeeze(args[0](args[1]))).numpy()

qs = [cirq.GridQubit(0, i) for i in range(2)]
X_train, X_test, y_train, y_test = encode(circle_data, circle_labels, qs)
sizes = [2, 8, 16]
num_rep = 3
toler = 1e-4
depth = 4
max_iter = 1000

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
