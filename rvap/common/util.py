#
# This module offers some utility functions for the VAP project.
#

import struct

BYTE_ORDER = 'little'

#
# Int16 -> Byte
#

def conv_2int16_2_byte(val1, val2):
    
    b1 = val1.to_bytes(2, BYTE_ORDER)
    b2 = val2.to_bytes(2, BYTE_ORDER)
    
    # print(b1)
    # print(b2)
    # concatenate two bytes
    b = b1 + b2
    
    #print(b)
    
    return b

def conv_2int16array_2_bytearray(arr1, arr2):
    
    if len(arr1) != len(arr2):
        raise ValueError('Two arrays must have the same length')
    
    b = b''
    
    for i in range(len(arr1)):
        b += conv_2int16_2_byte(int(arr1[i]), int(arr2[i]))
    
    return b

#
# Float32 -> Byte
#

def conv_2float_2_byte(val1, val2):
    
    b1 = struct.pack('<d', val1)
    b2 = struct.pack('<d', val2)
    
    b = b1 + b2
    
    return b

def conv_2floatarray_2_bytearray(arr1, arr2):
    
    if len(arr1) != len(arr2):
        raise ValueError('Two arrays must have the same length')
    
    b = b''
    
    for i in range(len(arr1)):
        b += conv_2float_2_byte(arr1[i], arr2[i])
    
    return b

def conv_float32_2_byte(val1, val2):
    
    b1 = struct.pack('<d', val1)
    b2 = struct.pack('<d', val2)

    b = b1 + b2
    
    return b

def conv_floatarray_2_byte(arr):
    
    b = b''
    
    for i in range(len(arr)):
        b += struct.pack('<d', arr[i])
    
    return b

#
# Byte -> Float32
#

def conv_byte_2_2float(b1, b2):
    
    val1 = struct.unpack('<d', b1)[0]
    val2 = struct.unpack('<d', b2)[0]
    
    return val1, val2

def conv_bytearray_2_2floatarray(barr):
    
    arr1, arr2 = [], []
    
    for i in range(0, len(barr), 16):
        b1 = barr[i:i+8]
        b2 = barr[i+8:i+16]
        
        val1, val2 = conv_byte_2_2float(b1, b2)
        
        arr1.append(val1)
        arr2.append(val2)
        
    return arr1, arr2

def conv_bytearray_2_floatarray(barr):
    
    arr = []
    
    for i in range(0, len(barr), 8):
        b = barr[i:i+8]
        val = struct.unpack('<d', b)[0]
        arr.append(val)
        
    return arr

#
# VAP result -> Byte
#
def conv_vapresult_2_bytearray(vap_result):
    
    b = b''
    #print(type(vap_result['t']))
    b += struct.pack('<d', vap_result['t'])
    
    b += len(vap_result['x1']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x1'])
    
    b += len(vap_result['x2']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x2'])
    
    b += len(vap_result['p_now']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_now'])

    b += len(vap_result['p_future']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_future'])
    
    return b

#
# Byte -> VAP result
#
def conv_bytearray_2_vapresult(barr):
    
    idx = 0
    t = struct.unpack('<d', barr[idx:8])[0]
    idx += 8
    
    len_x1 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x1 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x1])
    idx += 8*len_x1
    
    len_x2 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x2 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x2])
    idx += 8 * len_x2
    
    len_p_now = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_now = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_now])
    idx += 8*len_p_now
    
    len_p_future = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_future = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_future])

    result_vap = {
        't': t,
        'x1': x1,
        'x2': x2,
        'p_now': p_now,
        'p_future': p_future
    }
    
    return result_vap

#
# VAP result -> Byte
#
def conv_vapresult_2_bytearray_bc(vap_result):
    
    b = b''
    #print(type(vap_result['t']))
    b += struct.pack('<d', vap_result['t'])
    
    b += len(vap_result['x1']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x1'])
    
    b += len(vap_result['x2']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x2'])
    
    b += len(vap_result['p_bc_react']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_bc_react'])

    b += len(vap_result['p_bc_emo']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_bc_emo'])
    
    return b

#
# Byte -> VAP result
#
def conv_bytearray_2_vapresult_bc(barr):
    
    idx = 0
    t = struct.unpack('<d', barr[idx:8])[0]
    idx += 8
    
    len_x1 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x1 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x1])
    idx += 8*len_x1
    
    len_x2 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x2 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x2])
    idx += 8 * len_x2
    
    len_p_bc_react = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_bc_react = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_bc_react])
    idx += 8*len_p_bc_react
    
    len_p_bc_emo = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_bc_emo = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_bc_emo])

    result_vap = {
        't': t,
        'x1': x1,
        'x2': x2,
        'p_bc_react': p_bc_react,
        'p_bc_emo': p_bc_emo
    }
    
    return result_vap

if __name__ == '__main__':
    
    val1 = 15000
    val2 = 1
    
    b = conv_2int16_2_byte(val1, val2)
    print(len(b))
    
    import numpy as np
    
    arr1 = np.array([15000, 15000, 15000, 15000], dtype=np.int16)
    arr2 = np.array([1, 1, 1, 1], dtype=np.int16)
    
    barr = conv_2int16array_2_bytearray(arr1, arr2)
    print(len(barr))
    
    val1 = 1.0
    val2 = 1.0
    b = conv_2float_2_byte(val1, val2)
    print(len(b))
    
    arr1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    arr2 = np.array([1.0, 1.0, 3.0, 1.0], dtype=np.float32)
    
    barr = conv_2floatarray_2_bytearray(arr1, arr2)
    print(len(barr))
    
    arr1_, arr2_ = conv_bytearray_2_2floatarray(barr)
    print(arr1_)
    print(arr2_)
    