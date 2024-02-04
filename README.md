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
* [Acknowledgements](https://github.com/jjwyetzner/iQuHack#Acknowledgements)
* [Sources](https://github.com/jjwyetzner/iQuHack#Sources)

## Introduction
This project was designed to implement a Quantum Generative Adversarial Networks  (QGAN) based on the following [research paper](https://arxiv.org/pdf/2310.00585.pdf) by Wang et. Al. With the rise in advancements in integrated photonic technology, optical representations of quantum bits are becoming a promising alternative for quantum computation representation. In our project, we explore these photonic systems through the implementation of QGANs in the presence of noise, using stochastically adapted gradient descent to achieve higher fidelity. 

Consequently, we will present our own GAN machine learning model which utilizes a modified gradient descent technique. We hope that this GAN machine learning model can be used as a proof of concept about the possibility of performing adversarial training on a photonic chip. While the model is not useful for highly-meaningful tasks, in the future, such models can hopefully improve to higher degrees of utility.

To test the efficacy of GAN model, we trained a generator to produce the $\frac{1}{2}(\ket{01} + \ket{11} + \ket{23} + \ket{30})$ state from an input state which was the maximally entangled quqart $\frac{1}{2}(\ket{00} + \ket{11} + \ket{22} + \ket{33})$ through a QGAN photonic architecture (see [circuits](https://github.com/jjwyetzner/iQuHack#Circuits)). 

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
$\frac{1}{2}(\ket{00} + \ket{11} + \ket{22} + \ket{33})$ Our true state $\tau$ is given by $\frac{1}{2}(\ket{01} + \ket{12} + \ket{23} + \ket{30})$ The discriminator aims to rotate the true state and measure the $\ket{22}$ basis, which we can use as $M$ in our loss function. The generator and discriminator were modeled after the QGAN photonic architecture seen below from Wang et al.

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
For this process, we switched between training the discriminator and then the generator, each for roughly five-10 epochs at a time. By using this gradient descent with a cost function (the discriminator), we were able to converge to a local minimum. The diagram below illustrates our modified gradient descent process.

<center>
    <img src = "https://github.com/jjwyetzner/iQuHack/blob/main/images/gradientdescentdiagram.png">
</center>

## Results

## Bonus Quantum Heralding Approach
Since the state vectors used to prepare our quantumly entangled input state are not actually viable on Quandela’s hardware, we attempted to circumvent this through the use of quantum heralding. The goal of this approach was to achieve the end state of $\ket{1,0,0,0,1,0,0,0} + \ket{0,1,0,0,0,1,0,0}$, a superposition between the $\ket{00}$ and $\ket{11}$ equivalent state. First, we noted that we only needed to check the 0th, 1st, 4th, and 5th modes for 1-values, as the rest would all be equivalent to zero. To do this, we introduced two ancilla qubits, creating a 10-mode register as shown below. We applied beam splitters between the 0th and 1st modes as well as the 4th and 5th modes. The angle was chosen so that if there were a proton in either of the modes, it would go to the 0th and 4th modes, respectively. Then, to utilize the quantum heralding, we applied a permutation that routed the values of modes 0 and 8 to each other and 4 and 9 to each other. Modes 8 and 9 are required to have an input and output of 1, so this stage essentially guaranteed a photon in the 0th and 4th modes. From here, we then used a balanced beam-splitter between 0 and 1, and 4 and 5. This would redistribute the photon to either 0/1 and 4/5. The issue we encountered was the 50% probability that we would get $\ket{1,0,0,0,0,1,0,0}$ or $\ket{0,1,0,0,1,0,0,0}$, which we did not want. 

<center>
    <img src = "https://github.com/jjwyetzner/iQuHack/blob/main/images/quantumheraldingcircuit.png">
</center>

Our idea is to take away the second beam-splitter between 4/5  and to place CNOT gates on 4 and 5, both controlled by 1. This would ensure a 50/50 distribution of the two desired states, as seen above. However, we had trouble designing this type of gate in Perceval and moved on to focus on other parts of the project. 

## Bonus Noise Modeling
Wang, et. al characterize environmental noise using “the Gaussian noise model”, which is what we used to attempt to model noise. We added random noise to the phase shift weights each iteration by creating an array of random samples of a Gaussian distribution centered on the average weight value. It has a standard deviation of $\sigma = \sqrt{N} = \text{signal-to-noise ratio}$, and this mimics the effect of environmental noise in our system. It also affects our results accordingly, decreasing the model’s fidelity. We were unable to turn this into uncertainty values as we did not have time to train our model with noise. 

## Reflections
If we were to go back and do this project again, we would first focus on getting our model working and optimized as soon as possible. Our model performed extremely poorly when we first coded it, and it took hours to debug it and incrementally improve performance. One of the main consequences of this was that we found training for maximizing fidelity instead of minimizing loss yields significantly better results. Within 100 epochs, training against loss reached peak fidelity values of ~40%. In the same timeframe, training against fidelity easily reached peak values of 99.9%. We believe this is due to the inherently contrived nature of the QGAN setup, especially with the uncertainty within the discriminator circuit. The second aspect of our project we would put more time and focus on is understanding the discriminator. After maximizing our cost function from multiple initial states, we noted that some combinations of $\theta_D$ allowed the discriminator to prefer other superpositions that $\tau$ while still outputting high magnitudes for $\ket{22}$. This suggests that other states could also be rotated appropriately by our discriminator, weakening our model’s fidelity measure. A significant barrier we ran into during this project was a failure to understand the discriminator measurement. In Wang, et al., the suggested loss function is based on the second output port as a mathematical byproduct of changing bases to $P_{(2,2)}$. The discriminator does not fully rotate the $\tau$ vector into that basis though, which became a source of significant confusion. As a result of the partial rotation, the probability magnitude of $P_{(2,2)}$ reached a maximum value of $0.25$, a concerning weakness until we noticed that the paper contained a similar graph scale.

## Acknowledgements
We would like to thank Sam, Pierre, Marie, Alexia, and all of the other Quandela representatives who made this challenge possible for us today. We learned a lot of we are extremely appreciative. Thank you to iQuHack and everyone who helped create this academically enriching experience that we hope to take part in again! 

## Sources

Wang, Y., Xue, S., Wang, Y., Liu, Y., Ding, J., Shi, W., Wang, D., Liu, Y., Fu, X., Huang, G., Huang, A., Deng, M., & Wu, J. (2023). Quantum generative adversarial learning in photonics. Optics Letters, 48(20), 5197. https://doi.org/10.1364/ol.505084

Lloyd, S., & Weedbrook, C. (2018). Quantum Generative Adversarial Learning. Physical Review Letters, 121(4). https://doi.org/10.1103/physrevlett.121.040502 

Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K., McClean, J. R., Mitarai, K., Yuan, X., Cincio, Ł., & Coles, P. J. (2021). Variational quantum algorithms. Nature Reviews Physics, 3(9), 625–644. https://doi.org/10.1038/s42254-021-00348-9 

Karácsony, M., Oroszlány, L., & Zimborás, Z. (2023). Efficient qudit based scheme for photonic quantum computing. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2302.07357 

