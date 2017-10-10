import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import FloatProgress
from IPython.display import display

# S-curve parameters
SC_START = 0
SC_STOP = 1000
SC_STEP = 1
SC_ADD = 128

N_STEP = ((SC_STOP - SC_START)/SC_STEP) 
N_PIXELS = 2304

def read_scurve(sc_file_name):
    # display progress
    prog = FloatProgress(min = 0, max = N_STEP)
    display(prog)

    # read in and avg each DAC level 
    # NB: only necessary for earlier firmware versions (pre 01_08_2017)
    with open(sc_file_name, "rb") as f:
        sc_avg = []
        for i in range(N_STEP):
            byte = f.read(1)
            #while byte:
            data= []
            for j in range(N_PIXELS * SC_ADD): 
                byte = f.read(1)
                data.append(int(ord(byte)))
            dac_step = np.mean(np.reshape(data, (SC_ADD, N_PIXELS)), 0)
            sc_avg.append(dac_step)
            prog.value += 1

    print np.shape(dac_step)    
    print np.shape(sc_avg)
    return sc_avg
