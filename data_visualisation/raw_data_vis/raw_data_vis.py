from data_format import *
import numpy as np
from matplotlib import pyplot as plt
from euso_plotting_tools import *
from contextlib import contextmanager

class DataVis():
    """
    Readout raw CPU data files
    Input is filename of such a file
    """

    def __init__(self, filename):
        self.filename = filename

        # constants
        self._rows = 48
        self._cols = 48

        #initialise
        self.zynq_data_l1 = np.zeros((N_OF_FRAMES_L1_V0, N_OF_PIXEL_PER_PDM))
        self.zynq_data_l2 = np.zeros((N_OF_FRAMES_L1_V0, N_OF_PIXEL_PER_PDM))
        self.zynq_data_l3 = np.zeros((N_OF_FRAMES_L1_V0, N_OF_PIXEL_PER_PDM))
        self.scurve = np.zeros((NMAX_OF_THESHOLDS, N_OF_PIXEL_PER_PDM))
        
        self._file_type = None

        self.cpu_packet_num = 0
        
    def __enter__(self):
            return self

        
    def __exit__(self, exc_type, exc_val, exc_tb):
            self.filename = ""

            
    def _classify_file(self):
        """
        classify the file type 
        """
        if "CPU_RUN_MAIN" in self.filename:
            self._file_type = "cpu_main"
        elif "CPU_RUN_SC" in self.filename:
            self._file_type = "cpu_sc"
        elif "scurve" in self.filename:
            self._file_type = "raw_sc"
        else:
            print "ERROR: file type is not recognised"
            
    def _perform_mapping(self):
        """
        """
            
            
    def _read_data(self):

        DataVis._classify_file(self)
        
        if self._file_type == "cpu_main":
            
            with open(self.filename, "rb") as cpu_file:
                
                # move to the desired packet
                cpu_file.seek(sizeof(CpuFileHeader) * (self.cpu_packet_num + 1))
                packet = CPU_PACKET()
                size = cpu_file.readinto(packet)
            
                # put the zynq data into an indexed array
                for i in range(N_OF_FRAMES_L1_V0):
                    for j in range(N_OF_PIXEL_PER_PDM):
                        self.zynq_data_l1[i][j] = packet.zynq_packet.level1_data[self.cpu_packet_num].payload.raw_data[i][j]
                        self.zynq_data_l2[i][j] = packet.zynq_packet.level2_data[self.cpu_packet_num].payload.int16_data[i][j]
                        self.zynq_data_l3[i][j] = packet.zynq_packet.level3_data.payload.int32_data[i][j]

        elif self._file_type == "raw_sc":

            with open(self.filename, "rb") as sc_file:

                raw_scurve = DATA_TYPE_SCURVE_V1()
                size = sc_file.readinto(raw_scurve)

                # put the scurve data into an indexed array
                for i in range(NMAX_OF_THESHOLDS):
                    for j in range(N_OF_PIXEL_PER_PDM):     
                        self.scurve[i][j] = raw_scurve.int32_data[i][j]                       

        elif self._file_type == "cpu_sc":

            with open(self.filename, "rb") as sc_file:

                # move to the first packet
                sc_file.seek(sizeof(CpuFileHeader))
                scurve_packet = SC_PACKET()
                size = sc_file.readinto(scurve_packet)
                
                # put the scurve data into an indexed array
                for i in range(NMAX_OF_THESHOLDS):
                    for j in range(N_OF_PIXEL_PER_PDM):     
                        self.scurve[i][j] = scurve_packet.sc_data.payload.int32_data[i][j]                       

            
    def plot_pdm(self, level, gtu_num):

        DataVis._read_data(self)
        
        # get correct level of data
        if level == 1:
            fs = self.zynq_data_l1[gtu_num].reshape(self._rows, self._cols)
        elif level == 2:
            fs = self.zynq_data_l2[gtu_num].reshape(self._rows, self._cols)
        elif level == 3:
            fs = self.zynq_data_l3[gtu_num].reshape(self._rows, self._cols)
        else:
            print "ERROR: level not recognised"

        # make the plot    
        plot_focal_surface(fs)


    def plot_sc_3d(self):
        from mpl_toolkits.mplot3d import Axes3D

        DataVis._read_data(self)

        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111, projection='3d')
        nx, ny = NMAX_OF_THESHOLDS, N_OF_PIXEL_PER_PDM
        x = range(nx)
        y = range(ny)
    
        X, Y = np.meshgrid(x, y)
        
        ax.plot_surface(X, Y, np.transpose(self.scurve), cmap = 'viridis')

    def plot_sc_2d(self, dac_level):

        DataVis._read_data(self)

        fig = plt.figure(figsize = (10, 10))
        sc_2d = self.scurve[dac_level][:].reshape(self._rows, self._cols)
        plt.imshow(sc_2d)
