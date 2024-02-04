#JJ's circuit testing
import perceval as pcvl
import perceval.components as comp
from perceval.rendering.circuit import SymbSkin, PhysSkin
import math
import random


phiList = [(random.random() * math.pi) for x in range(6)]
theta = math.pi / 2

discriminator = pcvl.Circuit(4)

discriminator.add(0, comp.PS(phiList[0]))
discriminator.add(1, comp.PS(phiList[1]))
discriminator.add(2, comp.PS(phiList[2]))

discriminator.add((0, 1), comp.BS(theta))
discriminator.add((2, 3), comp.BS(theta))

discriminator.add(0, comp.PS(phiList[3]))
discriminator.add(2, comp.PS(phiList[4]))

discriminator.add((0, 1), comp.BS(theta))
discriminator.add((2, 3), comp.BS(theta))

discriminator.add((1, 2), comp.BS(theta))

discriminator.add(1, comp.PS(phiList[5]))

discriminator.add((1, 2), comp.BS(theta))

pcvl.pdisplay(discriminator, skin=SymbSkin())

discriminator.describe()