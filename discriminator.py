#JJ's circuit testing
import perceval as pcvl
import perceval.components as comp
import math
from perceval.rendering.circuit import SymbSkin, PhysSkin
import perceval

phi = math.pi / 2
theta = math.pi / 2

discriminator = pcvl.Circuit(4)

discriminator.add(0, comp.PS(phi))
discriminator.add(1, comp.PS(phi))
discriminator.add(2, comp.PS(phi))

discriminator.add((0, 1), comp.BS(theta))
discriminator.add((2, 3), comp.BS(theta))

discriminator.add(0, comp.PS(phi))
discriminator.add(2, comp.PS(phi))

discriminator.add((0, 1), comp.BS(theta))
discriminator.add((2, 3), comp.BS(theta))

discriminator.add((1, 2), comp.BS(theta))

discriminator.add(1, comp.PS(phi))

discriminator.add((1, 2), comp.BS(theta))

perceval.pdisplay(discriminator, skin=SymbSkin())

discriminator.describe()