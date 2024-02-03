#all imports

import perceval as pcvl
import math
import numpy as np

#given initial values and functions

d_theta = math.pi / 100
def discriminator():
  return None
epochs = None
phase_shiftor_angles = np.array([])

#training the phase_shiftor_angles

def training_generator(phase_shiftor_angles, epochs, d_theta, discriminator):
  for epoch in range(0, epochs):
    gradient = np.array([])
    for index in range(len(phase_shiftor_angles)):
      theta_plus = phase_shiftor_angles
      theta_plus[index] = phase_shiftor_angles[index] + d_theta
      theta_minus = phase_shiftor_angles
      theta_minus[index] = phase_shiftor_angles[index] - d_theta
      slope = (discriminator(theta_plus) - discriminator(theta_minus)) / (2*d_theta)
      gradient.append(slope)
    phase_shift_angles = phase_shiftor_angles - gradient * discriminator(phase_shiftor_angles)
