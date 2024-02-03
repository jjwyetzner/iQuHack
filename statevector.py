import perceval as pcvl
import random
from perceval import StateGenerator as stg
import exqalibur as exq

st0 = pcvl.StateVector("|0,0>")
st1 = pcvl.StateVector("|1,1>")
st2 = pcvl.StateVector("|2,2>")
st3 = pcvl.StateVector("|3,3>")

entangledvec = st0 + st1 + st2 + st3
print(entangledvec)