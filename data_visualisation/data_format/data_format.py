from ctypes import *
from .pdmdata import *

__all__ = ['CpuFileHeader', 'CpuFileTrailer', 'CpuPktHeader', 'CpuTimeStamp',
           'THERM_PACKET', 'HK_PACKET', 'ZYNQ_PACKET', 'CPU_PACKET', 'SC_PACKET',
           'N_CHANNELS_PHOTODIODE', 'N_CHANNELS_SIPM', 'N_CHANNELS_THERM', 'MAX_PACKETS_L1',
           'MAX_PACKETS_L2', 'MAX_PACKETS_L3', 'N_OF_FRAMES_L1_V0', 'RUN_SIZE']

#define constants
# for the analog readout 
N_CHANNELS_PHOTODIODE = 4
N_CHANNELS_SIPM = 64
N_CHANNELS_THERM = 10

# size of the zynq packets 
MAX_PACKETS_L1 = 4
MAX_PACKETS_L2 = 4
MAX_PACKETS_L3 = 1
N_OF_FRAMES_L1_V0 = 128

# number of packets in one run file
RUN_SIZE = 25

#define structures
class CpuFileHeader(Structure):
    _fields_ = [
        ("spacer", c_uint32),
        ("header", c_uint32),
        ("run_size", c_uint32),
    ]
    _pack_ = 1
    
class CpuFileTrailer(Structure):
    _fields_ = [
        ("spacer", c_uint32),
        ("run_size", c_uint32),
        ("crc", c_uint32),
    ]
    _pack_ = 1
    
class CpuPktHeader(Structure):
    _fields_ = [
        ("spacer", c_uint32),
        ("header", c_uint32),
        ("pkt_size", c_uint32),
        ("pkt_num", c_uint32),
    ]
    _pack_ = 1
    
class CpuTimeStamp(Structure):
    _fields_ = [
        ("cpu_time_stamp", c_uint32),
    ]   
    _pack_ = 1
    
class THERM_PACKET(Structure):
    _fields_ = [
        ("therm_packet_header", CpuPktHeader),
        ("therm_time", CpuTimeStamp),
        ("therm_data", c_float * N_CHANNELS_THERM),
    ]
    _pack_ = 1
    
class HK_PACKET(Structure):
    _fields_ = [
        ("hk_packet_header", CpuPktHeader),
        ("hk_time", CpuTimeStamp),
        ("photodiode_data", c_float * N_CHANNELS_PHOTODIODE),
        ("sipm_data", c_float * N_CHANNELS_SIPM),
        ("sipm_single", c_float),
    ]
    _pack_ = 1

class ZYNQ_PACKET(Structure):
    _fields_ = [
        ("N1", c_uint8),
        ("N2", c_uint8),
        ("level1_data", Z_DATA_TYPE_SCI_L1_V1 * MAX_PACKETS_L1),
        ("level2_data", Z_DATA_TYPE_SCI_L2_V1 * MAX_PACKETS_L2),
        ("level3_data", Z_DATA_TYPE_SCI_L3_V1),        
    ]
    pack_ = 1

class CPU_PACKET(Structure):
    _fields_ = [
        ("cpu_packet_header", CpuPktHeader),
        ("cpu_time", CpuTimeStamp),
        ("hk_packet", HK_PACKET),
        ("zynq_packet", ZYNQ_PACKET),
    ]
    _pack_ = 1

class SC_PACKET(Structure):
    _fields_ = [
        ("sc_packet_header", CpuPktHeader),
        ("sc_time", CpuTimeStamp),
        ("sc_start", c_uint16),
        ("sc_step", c_uint16),
        ("sc_stop", c_uint16),
        ("sc_acc", c_uint16),
        ("sc_data", Z_DATA_TYPE_SCURVE_V1),
    ]
    _pack_ = 1
    
        
    
