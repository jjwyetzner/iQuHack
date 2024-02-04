import perceval as pcvl
import math
import perceval.components as comp
from perceval.rendering.circuit import SymbSkin, PhysSkin

phi = math.pi/2

generator = pcvl.Circuit(4)

for i in range(1,4):
    generator.add(i, comp.PS(phi))
generator.add(1, comp.BS())
generator.add(1, comp.PS(phi))
generator.add(1, comp.BS())
generator.add(0, comp.BS())
generator.add(2, comp.BS())
generator.add(0, comp.PS(phi))
generator.add(0, comp.BS())
generator.add(2, comp.PS(phi))
generator.add(2, comp.BS())
generator.add(1, comp.PS(phi))
generator.add(1, comp.BS())
generator.add(1, comp.PS(phi))
generator.add(1, comp.BS())
generator.add(0, comp.PS(phi))
generator.add(2, comp.PS(phi))
generator.add(0, comp.BS())
generator.add(2, comp.BS())
generator.add(0, comp.PS(phi))
generator.add(0, comp.BS())
generator.add(2, comp.PS(phi))
generator.add(2, comp.BS())
generator.add(0, comp.PS(phi))
generator.add(1, comp.PS(phi))
generator.add(2, comp.PS(phi))


pcvl.pdisplay(generator, recursive=True, skin=SymbSkin())
