# QVC Optimizer Review 

Code for the paper "An Empirical Review of Optimization Techniques for Quantum Variational Circuits". 

Each of the python files can be run and will generate the results for the associated set of problem setups. To enable/disable noise, comment out or uncomment the only line that has the `.with_noise(cirq.depolarize(p=0.01))`. 

All the results are available in the `results.txt` organized by task, score, and standard deviation over the 3 runs. 

## Open Call For Research

Check out the [CONTRIBUTING.md](https://github.com/lockwo/qvc_opt_review/blob/main/CONTRIBUTING.md) and take on one of the issues (or make your own). For more information, see the video: https://www.youtube.com/watch?v=JSG5zSuSyKA

## Funding

This work was generoulsy supported by [Spell](https://spell.ml/) and their [Open Research Grant](https://spell.ml/blog/full-grant-recipient-YPcM3hUAACUAy4px)
