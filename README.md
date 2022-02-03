# QVC Optimizer Review 

Code for the paper "An Empirical Review of Optimization Techniques for Quantum Variational Circuits". 

Each of the python files can be run and will generate the results for the associated set of problem setups. To enable/disable noise, comment out or uncomment the only line that has the `.with_noise(cirq.depolarize(p=0.01))`. 

All the results are available in the `results.txt` organized by task, score, and standard deviation over the 3 runs. 
