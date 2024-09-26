# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:37:14 2024

@author: canal
"""

import csv
from os import listdir
from os.path import isfile, join
import numpy as np

def LoadFile(p, folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    data = []
    for file in onlyfiles:
        if p in file and "raw" in file.lower():
            with open(folder + file, newline='') as _file:
                data = list(csv.reader(_file))
            print("File loaded: ", file.split("/")[-1])
            break
    return data


if __name__ == "__main__":
    header_raw = ["gazeTheta", "gazePhi", "p1RotationX", "p1RotationY", "p1RotationZ", "p1RotationW", "p1PositionX", "p1PositionY", "p1PositionZ",
                  "p2RotationX", "p2RotationY", "p2RotationZ", "p2RotationW", "p2PositionX", "p2PositionY", "p2PositionZ", 
                  "relativeIntensity", "relativePitch", "isBlink", "blinkDuration", "isSaccade", "sampleConfidence"]
    
    
    header_transformed = ["gazeTheta", "gazePhi", "p1Theta", "p1Phi", "p2Theta", "p2Phi", 
                          "relativeIntensity", "relativePitch", "isBlink", "blinkDuration", "isSaccade", "sampleConfidence"]
    
    prefixes = ["2021_12_01_01", "2021_12_03_01", "2021_12_03_02", 
                "2021_12_03_03", "2021_12_03_05", "2021_12_07_01", 
                "2021_12_07_02", "2021_12_08_01", "2021_12_08_02", 
                "2021_12_08_03", "2021_12_08_04"]
    
    # phi_offset = [-0.06, -0.09] # for 07_01 and 08_03
    phi_offsets = [0, -9, -3, -2, -3, 3, 6, -10, -12, 10, 17, 3]
    
    i = 0
    total_length = 0
    for p in prefixes:
        
        # load new version of file
        folder = "./"
        new_data = LoadFile(p, folder)
        
        # load old version of file
        folder = "../Scripts/graphs/60fps/Original/"
        old_data = LoadFile(p, folder)
        
        output_data_original = []
        output_data_original.append(header_raw.copy())
        output_data_smoothed = []
        output_data_smoothed.append(header_raw.copy())
        
        suffix = str(i) if i > 9 else "0" + str(i)
        output_original_filename = "conv" + suffix + "_original.csv"
        output_processed_filename = "conv" + suffix + "_smoothed.csv"
        
        # if len(new_data) != len(old_data):
        #     print("data lengths do not match:", p)
            
        output_data_smoothed.extend(new_data[1:])
        
        # check column 7 (index 6) to get offset
        offset = False
        off_j = off_k = 0
        for j in range(20, len(new_data)):
            for k in range(j-20, len(old_data)):
                if new_data[j][6] == old_data[k][6]:
                    offset = True
                    off_j = j
                    off_k = k
                    break
            if offset:
                break

        # print("offset j:", off_j)
        # print("offset k:", off_k)
        offset = abs(off_j - off_k)
        
        if off_j < off_k:
            for j in range(len(old_data)-offset):
                new_row = new_data[j].copy()
                new_row[0] = old_data[j+offset][0]
                new_row[1] = str(float(old_data[j+offset][1]) + np.deg2rad(phi_offsets[i]))
                output_data_original.append(new_row)
                
        else:
            for j in range(offset,len(new_data)):
                new_row = new_data[j].copy()
                new_row[0] = old_data[j-offset][0]
                new_row[1] = str(float(old_data[j-offset][1]) + np.deg2rad(phi_offsets[i]))
                output_data_original.append(new_row)
            
        if len(output_data_original) - 1 != len(old_data):
            print("original lenth mismatch")
        if len(output_data_smoothed) != len(new_data):
            print("smoothed length mismatch")
            
        
        total_length += len(new_data) - (10*60)
        # for row in output_data_original:
        #     row.pop(20)
        # for row in output_data_smoothed:
        #     row.pop(20)
            
        # with open(output_original_filename, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(output_data_original)
            
        # with open(output_processed_filename, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(output_data_smoothed)
        
        # np.deg2rad(gaze_phi_offset)
        i += 1
        
    print("total data length:", total_length, total_length/60)
        
        
        
        
        
        