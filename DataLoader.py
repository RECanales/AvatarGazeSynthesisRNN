# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:09:12 2021

@author: rcanale

Handles loading head, eye, and audio data. Head and eye motion are loaded directly, AudioLoader is used to handle audio.
- Normalize head and eye movement
- Augment data
- Provide access to data
- carryover last valid frame during blinks
- maybe discard invalid data
"""

import numpy as np
import csv
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.signal import savgol_filter # smooth target position (z)
from pyquaternion import Quaternion

class DataLoader():
    def __init__(self):
        pass
    
    def To2DGazePoint(self, theta, phi):
        rho = 1;
        _x = rho * np.sin(phi) * np.cos(theta);
        _y = rho * np.cos(phi);
        gazepoint = np.array([_x, _y])  # 2d gazepoint
        return gazepoint
        
    def Audio2Binary(self, input_sequence):
        # replace audio features with 0 or 1, with 1 meaning there are audio features (speaking)
        audio_presence = np.reshape(np.array((np.abs(input_sequence[:,-2]) > 0)).astype(float), (len(input_sequence), 1))
        return np.concatenate((input_sequence[:,:-2], audio_presence), axis=1)
    
    
    def Standardize(self, data, means, sds):
        for i in range(len(data)):
            convo = self.Audio2Binary(data[i])
            # convo = data[i]
            # plt.scatter(range(len(convo)), convo[:,-1])
            # plt.show()
            for j in range(len(means)):
                convo[:,j] = (convo[:,j] - means[j]) / sds[j]
            data[i] = convo
        return data
        
        
    def LoadTrainingData(self, filepaths, framerate, lookback, which_features="all", shuffle=False, seed=123, pred_stride=1):
        np.random.seed(seed)
        print("seed", seed)
        
        # initiate lists for train, dev, test datasets
        # these store conversations seperately
        x_train = []
        y_train = []
        x_dev = []
        y_dev = []
        x_test = []
        y_test = []
        
        # store all features together for computing statistics
        # these store conversations together
        x_train_combined = []
        y_train_combined = []
        x_dev_combined = []
        x_test_combined = []
        
        # keep track of total number of train/dev/test examples
        num_train_examples = 0
        num_dev_examples = 0
        num_test_examples = 0
        total_count = 0
        bad_train_samples = 0
        bad_dev_samples = 0
        audio_samples = 0
        dev_audio_samples = 0
        
        train_percent = 0.6 # % of data used for training
        
        # TODO: store each conversation as a set of own windows, use nested loop while training
        dev_indices_prev = []
        test_indices_prev = []
        mirroredIndex = 0
        for filepath in filepaths:
                
            # load smoothed data from training graphs python script
            data = []
            with open(filepath, newline='') as _file:
                data = list(csv.reader(_file))
            # trim data
            trim_length = int(5 * framerate) # remove 5 seconds from end
            skip_length = int(5 * framerate) # remove 5 seconds from beginning
            data = data[skip_length:len(data) - trim_length]
            print("Start index:", skip_length, "End index:", len(data)-1, "Trimmed length (s):", len(data)/framerate)
            
            
            # Add displacement magnitude as feature
            new_data = []
            num_disp_frames = 1
            for i in range(num_disp_frames):
                new_data.append(data[i])
                new_data[i].insert(6, 0)
            
            for i in range(num_disp_frames, len(data)):
                theta_magnitude = np.square(float(data[i][2]) - float(data[i-num_disp_frames][2]))
                phi_magnitude = np.square(float(data[i][3]) - float(data[i-num_disp_frames][3]))
                disp_magnitude = theta_magnitude + phi_magnitude
                new_row = data[i]
                new_row.insert(6, disp_magnitude)
                new_data.append(new_row)
                
            data = new_data
            
            # split data
            data = np.array(data).astype(float)
            features = data[:,2:9]
            # print(features[0])
            if which_features == "motion":
                features = data[:,2:6]
            outputs = data[:,:2]
            blink_labels = np.expand_dims(data[:,-4], 1)
            confidences = np.expand_dims(data[:,-1], 1)
            saccade_labels = np.expand_dims(data[:,-2], 1)
            outputs = np.hstack((outputs, blink_labels, confidences, saccade_labels))
            
            train_len = int(train_percent * len(features)) # train on less data. previously 0.9
            dev_len = int(0.5 * (len(features) - train_len))
            test_len = len(features) - (train_len + dev_len)
            num_train_examples += train_len
            
            if "mirroredTheta" not in filepath:
                num_dev_examples += dev_len
                num_test_examples += test_len
            total_count += len(features)
            
            # lists for storing data from one conversation
            _x_train = []
            _y_train = []
            _x_dev = []
            _y_dev = []
            _x_test = []
            _y_test = []
                
            
            if shuffle == True:
                
                # randomize where train/dev/test set indices are
                dev_start_idx = np.random.randint(0, len(data)-dev_len)
                dev_end_idx = dev_start_idx + dev_len
                 
                test_start_idx = np.random.randint(0, len(data)-test_len)
                test_end_idx = test_start_idx + test_len
                
                if "mirroredTheta" not in filepath:
                    c = 0
                    while test_start_idx >= (dev_start_idx-test_len) and test_end_idx <= (dev_end_idx+test_len):
                        test_start_idx = np.random.randint(0, len(data)-test_len)
                        test_end_idx = test_start_idx + test_len
                        c += 1
                        if c > 1000:
                            print("Terminating shuffle early. Verify indices.")
                            break
                        
                    dev_indices_prev.append([dev_start_idx, dev_end_idx])
                    test_indices_prev.append([test_start_idx, test_end_idx])
                        
                else:
                    if mirroredIndex == 0:
                        print("\n\n")
                    dev_start_idx = dev_indices_prev[mirroredIndex][0]
                    dev_end_idx = dev_indices_prev[mirroredIndex][1]
                    test_start_idx = test_indices_prev[mirroredIndex][0]
                    test_end_idx = test_indices_prev[mirroredIndex][1]
                    mirroredIndex+=1
                    
                
                dev_indices = range(dev_start_idx, dev_end_idx)
                test_indices = range(test_start_idx, test_end_idx)
                for i in range(0, len(features)):
                    if i not in dev_indices and i not in test_indices:
                        _x_train.append(features[i])
                        _y_train.append(outputs[i])
                        x_train_combined.append(features[i])
                        y_train_combined.append(outputs[i])
                        
                if "mirroredTheta" not in filepath:
                    
                    for i in dev_indices:
                        _x_dev.append(features[i])
                        _y_dev.append(outputs[i])
                        x_dev_combined.append(features[i])
                        
                    for i in test_indices:
                        _x_test.append(features[i])
                        _y_test.append(outputs[i])
                        x_test_combined.append(features[i])
                    
                print("Train length:", len(_x_train), "Dev start index:", dev_start_idx, "Dev end index:", dev_end_idx, \
                      "Test start index:", test_start_idx, "Test end index:", test_end_idx, ", Total length", len(_x_train)+len(_x_dev)+len(_x_test))
                    
            else:
                for i in range(0, train_len):
                    _x_train.append(features[i])
                    _y_train.append(outputs[i])
                    x_train_combined.append(features[i])
                    
                if "mirroredTheta" not in filepath:
                    
                    for i in range(train_len, train_len + dev_len):
                        _x_dev.append(features[i])
                        _y_dev.append(outputs[i])
                        x_dev_combined.append(features[i])
                        
                    for i in range(train_len + dev_len, len(features)):
                        _x_test.append(features[i])
                        _y_test.append(outputs[i])
                        x_test_combined.append(features[i])
                    
                print("Dev index:", train_len, "Test index:", len(data) - test_len, ", Total length", len(_x_train)+len(_x_dev)+len(_x_test))
                
            _y_train = np.array(_y_train)
            _y_dev = np.array(_y_dev)
            _x_train = np.array(_x_train)
            x_train.append(_x_train)
            y_train.append(_y_train)
            
            if "mirroredTheta" not in filepath:
                _x_dev = np.array(_x_dev)
                x_dev.append(_x_dev)
                y_dev.append(np.array(_y_dev))
                x_test.append(np.array(_x_test))
                y_test.append(np.array(_y_test))
                bad_dev_samples += np.sum(_y_dev[:,-2]<0.6)
                dev_audio_samples += np.sum(np.absolute(_x_dev[:,-2]) > 0)
                
            bad_train_samples += np.sum(_y_train[:,-2]<0.6)
            audio_samples += np.sum(np.absolute(_x_train[:,-2]) > 0)
            
        # ragged sequences
        print("Number of conversations:", len(x_train))
        print("Total number of training examples:", num_train_examples)
        print("Total number of dev/test examples:", num_dev_examples, num_test_examples)
        print("Percentage of bad samples for training (conf < 0.6):", round(100*(bad_train_samples/num_train_examples),3))
        print("Percentage of bad samples in dev set (conf < 0.6):", round(100*(bad_dev_samples/num_dev_examples),3))
        print("Percent of training set with audio samples:", round(100 * (audio_samples/num_train_examples), 3))
        print("Percent of dev set with audio samples:", round(100 * (dev_audio_samples/num_dev_examples), 3))
        y_train_combined = np.array(y_train_combined)
        print("Mean training set confidence:", np.mean(y_train_combined[:,-2]))
        
        # self.DatasetAnalysis(x_train_combined)
        
        # get means, standard deviations, min-max scale and center
        means = []
        sds = []
        x_train_combined = np.array(x_train_combined)
        x_dev_combined = np.array(x_dev_combined)
        x_test_combined = np.array(x_test_combined)
        print("Before:", x_train_combined.shape)
        
        
        # Replace audio features with presence of audio (0 or 1)
        x_train_combined = self.Audio2Binary(x_train_combined)
        print("After:", x_train_combined.shape)
        
        for i in range(x_train_combined.shape[1]):
            mu = np.mean(x_train_combined[:,i])
            sd = np.std(x_train_combined[:,i])
            means.append(mu)
            sds.append(sd)
        
        x_train_combined = []
        x_dev_combined = []
        x_test_combined = []
        
        # plt.scatter(range(len(x_train[0])), x_train[0][:,-1])
        # plt.show()
        x_train = self.Standardize(x_train, means, sds)
        x_dev = self.Standardize(x_dev, means, sds)
        x_test = self.Standardize(x_test, means, sds)
            
        # extract windows of data (cast to float32)
        window_size = framerate * lookback # framerate * lookback (s)
        stride = pred_stride # TODO: 1 for predicting single output
        train_x_windows = []
        test_x_windows = []
        dev_x_windows = []
        dev_y_windows = []
        train_y_windows = []
        test_y_windows = []
        
        # plt.scatter(range(len(x_train[0])), x_train[0][:,-1])
        # plt.show()
        
        # break sequences (conversations) into subsequences of length 'window_size'
        # would they overlap?
        train_length = 0
        for i in range(len(x_train)):
            _train_x_windows = []
            _train_y_windows = []
            for j in range(window_size, len(x_train[i]), stride):
                _train_x_windows.append(x_train[i][j-window_size:j])
                _train_y_windows.append(y_train[i][j-window_size:j])
                
            _train_x_windows = np.array(_train_x_windows)
            _train_y_windows = np.array(_train_y_windows)
            # print("Train feature windows shape:", _train_x_windows.shape)
            train_x_windows.append(_train_x_windows)
            train_y_windows.append(_train_y_windows)
            train_length += len(_train_x_windows)
            
        dev_length = 0
        for i in range(len(x_dev)):
            _dev_x_windows = []
            _dev_y_windows = []
            for j in range(window_size, len(x_dev[i]), stride):
                _dev_x_windows.append(x_dev[i][j-window_size:j])
                _dev_y_windows.append(y_dev[i][j-window_size:j])
                
            _dev_x_windows = np.array(_dev_x_windows)
            _dev_y_windows = np.array(_dev_y_windows)
            # print("Dev feature windows shape:", _dev_x_windows.shape)
            dev_x_windows.append(_dev_x_windows)
            dev_y_windows.append(_dev_y_windows)
            dev_length += len(_dev_x_windows)
            
        # TODO review code. make sure dev/test sets are returned
        test_length = 0
        for i in range(len(x_test)):
            _test_x_windows = []
            _test_y_windows = []
            for j in range(window_size, len(x_test[i]), stride):
                _test_x_windows.append(x_test[i][j-window_size:j])
                _test_y_windows.append(y_test[i][j-window_size:j])
                
            _test_x_windows = np.array(_test_x_windows)
            _test_y_windows = np.array(_test_y_windows)
            # print("test feature windows shape:", _test_x_windows.shape)
            test_x_windows.append(_test_x_windows)
            test_y_windows.append(_test_y_windows)
            test_length += len(_test_x_windows)
                
        return train_x_windows, dev_x_windows, test_x_windows, \
            train_y_windows, dev_y_windows, test_y_windows, means, sds