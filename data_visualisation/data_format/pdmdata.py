from ctypes import *

#define constants
N_OF_PIXEL_PER_PDM = 2304

N_OF_FRAMES_L1_V0 = 128
N_OF_FRAMES_L2_V0 = 128
N_OF_FRAMES_L3_V0 = 128

NMAX_OF_THESHOLDS = 1024
PAD_VAL = 4294967295

class TimeStamp_dual(Structure):
    _fields_ = [
        ("n_gtu", c_uint32),
        ("unix_time", c_uint32),
    ]
    _pack_ = 1

class ZynqBoardHeader(Structure):
    _fields_ = [
        ("header", c_uint32),
        ("payload_size", c_uint32)
    ]
    _pack_ = 1

    
class DATA_TYPE_SCI_L1_V1(Structure):
    _fields_ = [
        ("ts", TimeStamp_dual),
        ("trig_type", c_uint32),
        ("cathode_status", c_uint8 * 12),
        ("raw_data", (c_uint8 * N_OF_PIXEL_PER_PDM) * N_OF_FRAMES_L1_V0),
    ]
    _pack_ = 1

class Z_DATA_TYPE_SCI_L1_V1(Structure):
    _fields_ = [
        ("zbh", ZynqBoardHeader),
        ("payload", DATA_TYPE_SCI_L1_V1),
    ]
    _pack_ = 1
    
class DATA_TYPE_SCI_L2_V1(Structure):
    _fields_ = [
        ("ts", TimeStamp_dual),
        ("trig_type", c_uint32),
        ("cathode_status", c_uint8 * 12),
        ("int16_data", (c_uint16 * N_OF_PIXEL_PER_PDM) * N_OF_FRAMES_L2_V0),
    ]
    _pack_ = 1
    
class Z_DATA_TYPE_SCI_L2_V1(Structure):
    _fields_ = [
        ("zbh", ZynqBoardHeader),
        ("payload", DATA_TYPE_SCI_L2_V1),
    ]
    _pack_ = 1

class DATA_TYPE_SCI_L3_V1(Structure):
    _fields_ = [
        ("ts", TimeStamp_dual),
        ("trig_type", c_uint32),
        ("cathode_status", c_uint8 * 12),
        ("hv_status", c_uint32),
        ("int32_data", (c_uint32 * N_OF_PIXEL_PER_PDM) * N_OF_FRAMES_L3_V0),
    ]
    _pack_ = 1

class Z_DATA_TYPE_SCI_L3_V1(Structure):
    _fields_ = [
        ("zbh", ZynqBoardHeader),
        ("payload", DATA_TYPE_SCI_L3_V1),
    ]
    _pack_ = 1

class DATA_TYPE_SCURVE_V1(Structure):
    _fields_ = [
        ("int32_data", (c_uint32 * N_OF_PIXEL_PER_PDM) * NMAX_OF_THESHOLDS),
    ]
    _pack_ = 1

class Z_DATA_TYPE_SCURVE_V1(Structure):
    _fields_ = [
        ("zbh", ZynqBoardHeader),
        ("payload", DATA_TYPE_SCURVE_V1),
    ]
    _pack_ = 1
    
    
