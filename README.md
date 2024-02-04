# Yale Varsity Clash Royale Team (Team 33) - iQuHack 2024, Quandela Challenge

## Table of Contents
* [Introduction](https://github.com/jjwyetzner/iQuHack#Introduction)
* [Dependencies](https://github.com/jjwyetzner/iQuHack#Dependencies)
* [Circuits](https://github.com/jjwyetzner/iQuHack#Circuits)
* [State Vectors](https://github.com/jjwyetzner/iQuHack#State-Vectors)
* [GAN Machine Learning Model](https://github.com/jjwyetzner/iQuHack#Gan-Machine-Learning-Model)
* [Results](https://github.com/jjwyetzner/iQuHack#Results)
* [Bonus Quantum Heralding Approach](https://github.com/jjwyetzner/iQuHack#Bonus-Quantum-Heralding-Approach)
* [Bonus Noise Modeling](https://github.com/jjwyetzner/iQuHack#Bonus-Noise-Modeling)
* [Reflections](https://github.com/jjwyetzner/iQuHack#Reflections)
* [Sources](https://github.com/jjwyetzner/iQuHack#Sources)

## Introduction
This project was designed to implement a Quantum Generative Adversarial Networks  (QGAN) based on the following [research paper](https://arxiv.org/pdf/2310.00585.pdf) by Wang et. Al. With the rise in advancements in integrated photonic technology, optical representations of quantum bits are becoming a promising alternative for quantum computation representation. In our project, we explore these photonic systems through the implementation of QGANs in the presence of noise, using stochastically adapted gradient descent to achieve higher fidelity. 

Consequently, we will present our own GAN machine learning model which utilizes [what does it utilize?]. We hope that this GAN machine learning model can be used as a proof of concept about the possibility of performing adversarial training on a photonic chip. While the model is not useful for highly-meaningful tasks, in the future, such models can hopefully improve to higher degrees of utility.

To test the efficacy of GAN model, we trained a generator to produce the $\frac{1}{2}(\ket{01} + \ket{11} + \ket{23} + \ket{30})$ state from an input state which was the maximally entangled quqart $\frac{1}{2}(\ket{00} + \ket{11} + \ket{22} + \ket{33})$ through a QGAN photonic architecture (see [circuits](https://github.com/jjwyetzner/iQuHack#Circuits)). 

[summary of actual final results of paper]


## Dependencies
To install all dependencies automatically run:
```
pip install -r requirements.txt
```

The following packages will be necessary for our QGAN model:
```
jupyter==1.0.0
matplotlib==3.8.2
numpy==1.26.3
perceval-quandela==0.10.3
```
## Circuits
Wang et al. ‘23 describe an elementary QGAN based around a two-player game. Both players utilize variational quantum circuits with trainable parameters. The first player trains a generator with parameters $\theta_G$, while his opponent trains a discriminator circuit with some parameters $\theta_D$. The goal of the generator is to take in a general quantum state and produces a candidate quantum state $\rho(\theta_G)$ that mimics a true state $\tau$. The discriminator’s role is to perform measurements and identify the true state from the generated state. The difference in the discriminator’s ability is identified by the function $d(\theta_G, \theta_D) = |M(\theta_D)\rho(\theta_G) - M(\theta_D)\tau|$ where $M$ represents the measurements made by the discriminator

In our model, the input state is the maximally-entangled state, 
$\frac{1}{2}(\ket{00} + \ket{11} + \ket{22} + \ket{33})$ Our true state $\tau$ is given by $\frac{1}{2}(\ket{01} + \ket{11} + \ket{23} + \ket{30})$ The discriminator aims to rotate the true state and measure the $\ket{22}$ basis, which we can use as $M$ in our loss function. The generator and discriminator were modeled after the QGAN photonic architecture seen below from Wang et al.

<center>
    <img src="https://github.com/jjwyetzner/iQuHack/blob/main/images/photonicQGAN.png">
</center>

The full circuit was modeled with the predefined beam splitters and phase shifters located in `perceval.components`, and the full circuit consisted of 8 mode inputs. The generator circuit that we created can be seen below.

Generator:
<center>
    <img src="https://github.com/jjwyetzner/iQuHack/blob/main/images/generator_fig.jpeg">
</center>

The discriminator that we created can be seen below.

Discriminator:
<center>
    <img src="https://github.com/jjwyetzner/iQuHack/blob/main/images/discriminator_fig.jpeg">
</center>

## State Vectors
Our first main implementation hurdle was figuring out a way to represent the data such that it can be passed through the circuit and modified as such. Our first solution was to use the StateVector class.

The `StateVector` class allows for superposition of basic states, so we represented the maximally entangled ququart $\alpha$, called `stf` in our code, as a superposition of four 8-mode states: 
$\alpha = \frac{1}{2}(\ket{00} + \ket{11} + \ket{22} + \ket{33}) = \frac{1}{2}(\ket{1,0,0,0,1,0,0,0} + \ket{0,1,0,0,0,1,0,0} + \ket{0,0,1,0,0,0,1,0} + \ket{0,0,0,1,0,0,0,1})$
This vector is passed into the generator circuit, which yields a superposition of four new 8-mode states, and this result is passed through the discriminator and sampled to give the distribution of states that are $\ket{22}$.
A similar procedure is performed for the target state vector $\tau$, which is passed directly into the discriminator and sampled in the same way. The absolute value of the differences of these two distributions is the loss, which is returned out of the `calculate_loss` function.

```python
def calculate_loss(generator_angles, discriminator_angles): #for minimizing from an ungenerated vector
 for i in range(30):
   gen_params[i].set_value(generator_angles[i])
 for i in range(12):
   discrim_params[i].set_value(discriminator_angles[i])
 g_sampler = generate(generator_angles, stf)
 g_sample = g_sampler.samples(1)['results'][0]
 d_sampler = discriminate(discriminator_angles, g_sample)
 dist1 = d_sampler.probs()['results'][pcvl.BasicState("|0, 1, 0, 0, 0, 1, 0, 0>")]


 tau_d_sampler = discriminate(discriminator_angles, tau)
 dist2 = tau_d_sampler.probs()['results'][pcvl.BasicState("|0, 1, 0, 0, 0, 1, 0, 0>")]


 return abs(dist1 - dist2), dist1, dist2
```


## GAN Machine Learning Model
To determine the appropriate phase shift parameters, we applied a modified gradient descent machine learning technique. Due to the stochastic nature of quantum phenomena, our function is non-differentiable, and therefore, the standard gradient descent is not possible. Instead, we used the Newton-Raphson method, and defined a discrete analog to the gradient descent for our version of backpropagation. We considered each angle of phase shifting to be the weights in our model. For each angle $\theta_n$, we calculated the slope of the secant line between points $\theta_n \pm d\theta$. The slope, $k_n$ is equal to:
$$\frac{d((\theta_d + d\theta), \theta_g) - d((\theta_d +- d\theta), \theta_g)}{2d\theta}$$
We equate the gradient, $\nabla d$ to the vertical of $k_n$. The generator and discriminator both utilize the same cost function, so we either subtract or add this gradient to minimize or maximize the value of $d$, respectively. For generating, we minimized, and for discriminating, we maximized. Therefore the new weights, $\theta’$ are defined as:
$$\theta’ = \theta \pm  k_nd\theta \nabla$$
For this process, we switched between training the discriminator and then the generator, each for roughly five-10 epochs at a time. By using this gradient descent with a cost function (the discriminator), we were able to converge to a local minimum.


## Results

## Bonus Quantum Heralding Approach

## Bonus Noise Modeling

## Reflections
If we were to go back and do this project again, we would first focus on getting our model working and optimized as soon as possible. Our model performed extremely poorly when we first coded it, and it took hours to debug it and incrementally improve performance. One of the main consequences of this was that we found training for maximizing fidelity instead of minimizing loss yields significantly better results. Within 100 epochs, training against loss reached peak fidelity values of ~40%. In the same timeframe, training against fidelity easily reached peak values of 99.9%. We believe this is due to the inherently contrived nature of the QGAN setup, especially with the uncertainty within the discriminator circuit. The second aspect of our project we would put more time and focus on is understanding the discriminator. After maximizing our cost function from multiple initial states, we noted that some combinations of $\theta_D$ allowed the discriminator to prefer other superpositions that $\tau$ while still outputting high magnitudes for $\ket{22}$. This suggests that other states could also be rotated appropriately by our discriminator, weakening our model’s fidelity measure. A significant barrier we ran into during this project was a failure to understand the discriminator measurement. In Wang, et al., the suggested loss function is based on the second output port as a mathematical byproduct of changing bases to $P_{(2,2)}$. The discriminator does not fully rotate the $\tau$ vector into that basis though, which became a source of significant confusion. As a result of the partial rotation, the probability magnitude of $P_{(2,2)}$ reached a maximum value of 0.25, a concerning weakness until we noticed that the paper contained a similar graph scale.



## Sources

