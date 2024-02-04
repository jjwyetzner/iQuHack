# Yale Varsity Clash Royale Team (Team 33) - iQuHack 2024, Quandela Challenge

## Table of Contents
* [Introduction](https://github.com/jjwyetzner/iQuHack#Introduction)
* [Dependencies](https://github.com/jjwyetzner/iQuHack#Dependencies)
* [Circuits](https://github.com/jjwyetzner/iQuHack#Circuits)
* [State Vectors](https://github.com/jjwyetzner/iQuHack#State-Vectors)
* [GAN Machine Learning Model](https://github.com/jjwyetzner/iQuHack#Gan-Machine-Learning-Model)
* [Results](https://github.com/jjwyetzner/iQuHack#Results)
* [Sources](https://github.com/jjwyetzner/iQuHack#Sources)
* [Reflections](https://github.com/jjwyetzner/iQuHack#Reflections)

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

## State Vectors

## GAN Machine Learning Model

## Results

## Reflections

## Sources
