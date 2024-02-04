import perceval as pcvl
from perceval import Circuit
from perceval.components import BS, PERM, Port
from perceval.utils import Encoding, PostSelect
import math
import numpy as np
import perceval.components as comp
from perceval.rendering.circuit import SymbSkin, PhysSkin

theta = math.pi/2
generator = pcvl.Circuit(10)
# # M = pcvl.Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                  [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
#                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
#                  [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
# generator.add((0,1), comp.PERM([0,1]))
generator.add(0, comp.BS(math.pi)) #send everything from 0 and 1 to 0
generator.add(4, comp.BS(math.pi)) #ditto
generator.add(0, comp.PERM([8,1,2,3,9,5,6,7,0,4]))
generator.add(0, comp.BS(theta = math.pi/2))
# generator.add(0, comp.Unitary(M))
# c2 = pcvl.Circuit.decomposition(M, comp.BS, shape="triangle")
# c2.describe()
generator.add(4, comp.BS(theta = math.pi/2))

st1 = pcvl.StateVector("|>")
st2 = pcvl.StateVector("|>")
st3 = pcvl.StateVector("|>")
st4 = pcvl.StateVector("|>")


generation_processor = pcvl.Processor("SLOS", generator)
generation_processor.add_herald(8, 1)
generation_processor.add_herald(9, 1)
# generation_processor.with_input(pcvl.StateVector("|1,0,0,0,1,0,0,0,1,1>") + pcvl.StateVector("|0,1,0,0,0,1,0,0,1,1>"))
# sampler = pcvl.algorithm.Sampler(generation_processor)
# print(sampler.probs()["results"])



pcvl.pdisplay(generator, skin=SymbSkin())
# pcvl.pdisplay(generator.U)
# print(generator.U)
