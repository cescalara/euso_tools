import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors, ticker, cm
import seaborn as sns
from euso_plotting_tools import *

class SimDataVis():
    """
    Read out and plot outputs of ESAF simulation
    Input is simulation file name (.txt or .root)
    """

    def __init__(self, filename):

        self.filename = filename

        # constants
        self._dt_min = 2.5e-6 # [s]
        self._rows = 48
        self._cols = 48
        self._n_pmt = 36
        self._n_pmt_rows = 8
        self._n_pmt_cols = 8

        #initialise
        self.data_level = 0
        self._file_type = None
        self.dt = self._dt_min # level 1 data default
        self.pdm_data = None
        self.light_curve = []
        self.n_gtu = 0
        
    def __enter__(self):
        return self

        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.filename = ""
        self.data_level = 0


    def _classify_file(self):
        """
        classify the input file type
        """
        if ".txt" in self.filename:
            self._file_type = "sim_text"
        elif ".root" in self.filename:
            self._file_type = "sim_root"
        else:
            print "ERROR: file type is not recognised"

            
    def set_data_level(self, data_level):
        """
        set the level of data to work with 
        """
        if data_level < 4 and data_level > 0:
            self.data_level = data_level
        else:
            print "ERROR: data level must be 1, 2 or 3"

            
    def _get_time_step(self):
        """
        get the time step to work with from the data level
        """
        if self.data_level == 1:
            self.dt = self._dt_min
        elif self.data_level == 2:
            self.dt = self._dt_min * 128
        elif self.data_level == 3:
            self.dt = self._dt_min * 128 * 128
        else:
            print "ERROR: data level must be set"

            
    def _read_data(self):
        """
        read out the data depending on the file type
        """

        SimDataVis._classify_file(self)
        SimDataVis._get_time_step(self)
        
        if self._file_type == "sim_text":

            with open(self.filename) as f:

                content = []
                for line in f:
                    line = line.split()
                    if line:
                        line = [int(i) for i in line]
                        content.append(line)
                        
                        len = np.shape(content)[0]
                        index = range(0, len, self._rows + 1)
                        #n_gtu = int(content[index[-1]][0]) 
                        self.n_gtu = (np.shape(index))[0] - 1
                        self.pdm_data = np.zeros((self.n_gtu, self._rows, self._cols))
                        t = np.arange(0, round(self.n_gtu * self.dt, 6), self.dt)
                        
                        for gtu_counter in range(self.n_gtu):
                            self.pdm_data[gtu_counter][:][:] = content[index[gtu_counter] + 1 : index[gtu_counter] + (self._rows + 1)]
                            self.light_curve.append(np.sum(self.pdm_data[gtu_counter][:][:]))
                            

        elif self._file_type == "sim_root":

            # Add ROOT file readout
            n = 0

        else:
            print "ERROR: input file unclassified"
                

    def plot_pdm_sum(self):
        """
        plot the sum of the file over all GTUs
        """

        SimDataVis._read_data(self)
        
        # get the sum from the data
        pdm_sum = np.sum(self.pdm_data, axis = 0)
        
        # make the plot
        plot_focal_surface(pdm_sum)

    
