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
        self._n_pmt = 36
        self._n_pmt_rows = 8
        self._n_pmt_cols = 8

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
            
    def _map_data(self, input_data):
        """
        map the data to show physical location of pixels on the PDM 
        input is a 1d vector of N_OF_PIXEL_PER_PDM pixels
        """

        # split into PMTs
        for i in range(self._n_pmt):
            for x in range(self._n_pmt_rows):
                for y in range(self._n_pmt_cols):
                    map_data[i][x][y] = input_data[(y + (x * 8)) + (i * 64)]

        # organise into columns and perform rotations
        col1 = [mapping[0], np.rot90(mapping[1], 1), mapping[2], 
                np.rot90(mapping[3], 1), mapping[4], np.rot90(mapping[5],1)]
    
        col2 = [np.rot90(mapping[6], 3), np.rot90(mapping[7], 2), np.rot90(mapping[8], 3),
                np.rot90(mapping[9], 2), np.rot90(mapping[10], 3), np.rot90(mapping[11], 2)]
    
        col3 = [mapping[12], np.rot90(mapping[13], 1), mapping[14], 
                np.rot90(mapping[15], 1), mapping[16], np.rot90(mapping[17], 1)]
            
        col4 = [np.rot90(mapping[18], 3), np.rot90(mapping[19], 2), np.rot90(mapping[20], 3), 
                np.rot90(mapping[21], 2), np.rot90(mapping[22], 3), np.rot90(mapping[23], 2)]
            
        col5 = [mapping[24], np.rot90(mapping[25], 1), mapping[26], 
                np.rot90(mapping[27], 1), mapping[28], np.rot90(mapping[29], 1)]
            
        col6 = [np.rot90(mapping[30], 3), np.rot90(mapping[31], 2), np.rot90(mapping[32], 3), 
                np.rot90(mapping[33], 2), np.rot90(mapping[34], 3), np.rot90(mapping[35], 2)]

        c1 = np.concatenate(col1, 0)
        c2 = np.concatenate(col2, 0)
        c3 = np.concatenate(col3, 0)
        c4 = np.concatenate(col4, 0)
        c5 = np.concatenate(col5, 0)
        c6 = np.concatenate(col6, 0)

        # rebuild PDM
        all_rows = [c1, c2, c3, c4, c5, c6]
        pdm = np.concatenate(all_rows, 1)

        return pdm
    
    def _read_data(self):
        """
        read the data from the file depending on the file type
        """

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
                        self.zynq_data_l1[i][j] =
                        packet.zynq_packet.level1_data[self.cpu_packet_num].payload.raw_data[i][j]
                        self.zynq_data_l2[i][j] =
                        packet.zynq_packet.level2_data[self.cpu_packet_num].payload.int16_data[i][j]
                        self.zynq_data_l3[i][j] =
                        packet.zynq_packet.level3_data.payload.int32_data[i][j]

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
        """
        plot the PDM
        input the level of data and GTU #
        """
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
        pdm = DataVis._map_data()
        plot_focal_surface(fs)


    def plot_sc_3d(self):
        """
        plot a 3d view of the Scurve
        """
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
        """
        plot a 2d view of the scurve for a given dac level
        """

        DataVis._read_data(self)

        fig = plt.figure(figsize = (10, 10))
        sc_2d = self.scurve[dac_level][:].reshape(self._rows, self._cols)
        plt.imshow(sc_2d)