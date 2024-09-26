# -*- coding: utf-8 -*-
"""
@author: Ryan Canales

Script for loading data and training GRU RNN from "Real-Time Conversational Gaze Synthesis for Avatars"
Link to paper (open access): https://dl.acm.org/doi/abs/10.1145/3623264.3624446
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import csv
from DataLoader import DataLoader as FeatureLoader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

from config import training_params, data_params, motion_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MotionGRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(MotionGRUNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.5)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.final_fc = nn.Linear(hidden_dim, 2) # was output dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, x, h, pred_frames):
        x = x.to(device)
        
        # FC1, FC2
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(self.fc3(x))
        x = self.relu(x)
        x = self.dropout(self.fc4(x))
        
        # recurrent module (GRU)
        out, h = self.gru(x, h) # batch, sequence length, hidden state size
        
        out = self.relu(out[:,-pred_frames:]) # multiple frame output
        
        out_eye = self.dropout(self.fc5(out))
        out_eye = self.relu(out_eye)
        out_eye = self.dropout(self.fc6(out_eye))
        out_eye = self.relu(out_eye)
        out_eye = self.final_fc(out_eye)
        
        return out_eye, h
    
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
    
class EyeMotionModel():
    def __init__(self, train_windows, audio_windows, dev_windows, test_windows, train_outputs, dev_outputs, test_outputs, framerate, audio=True):
        self.train_windows = train_windows
        self.test_windows = test_windows
        self.audio_windows = audio_windows
        self.dev_windows = dev_windows
        self.dev_outputs = dev_outputs
        # self.freqs = freqs
        # self.bins = bins
        
        self.outputs = train_outputs
        self.test_outputs = test_outputs
        
        self.framerate = framerate
        print("Device:", device)
        
        
        # self.output_dim = self.outputs.shape[-1] - 1 # -1 for blink
        self.output_dim = 2 # Spherical: theta, phi ; single frame 
        
        self.motion_input_dim = 6
        if audio == False:
            self.motion_input_dim = 5
        print("motion input dim:", self.motion_input_dim, "output dim:", self.output_dim)
        
        self.motion_layers = 3 # GRU layers
        
        self.motion_hidden_dim = 128
        
        self.training_output = []
        
    def DivideData(self, seed):
        self.seed = seed
        
        # split into training, testing sets (90, 10)       
        print("Feature lengths:", len(self.train_windows), len(self.audio_windows))
        feature_length = len(self.train_windows)
        
        # print("post:", np.mean(self.features[:,:,5]), np.std(self.features[:,:,5]))
        
        if len(self.audio_windows) == 0:
            print("No audio features loaded.")
        else:
            feature_length = int(min(len(self.train_windows), len(self.audio_windows)))
        
        self.train_windows = self.train_windows[:feature_length]
        print('Feature shape:', self.train_windows[0].shape)
        print("Feature length:", len(self.train_windows))
        
        if len(self.audio_windows) > 0:
            self.audio_windows = self.audio_windows[:feature_length]
            print("Feature length, trimmed:", len(self.train_windows), len(self.audio_windows))
        
        # testing unshuffled sequences
        self.train_x_motion = self.train_windows.copy()
        self.train_y = self.outputs.copy()
        self.test_x_motion = self.test_windows.copy()
        self.test_y = self.test_outputs.copy()
        
        self.train_windows = []
        self.test_windows = []
        self.audio_windows = []
        self.outputs = []
        self.test_outputs = []
        
        
    def SaveTrainingOutput(self, filename, data, append=False):
        mode = 'a' if append else 'w'
        with open(filename + ".csv", mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
            
    
    def TrainMotionRNN(self, lr=0.01, epochs=5, batch_size=256, step=1, model_name="", audio=True, pred_frames=1):
        # motion gru net
        self.motion_model = MotionGRUNet(6, self.motion_hidden_dim, self.output_dim, self.motion_layers).to(device)
        
        gaze_criterion = nn.MSELoss() # GRU only
        params = self.motion_model.parameters()
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
        
        features = []
        gaze_labels = []
        
        print("Training motion RNN...")
        cached_params = None
        
        # Start training loop
        # TODO make own batches, implement loss s.t. padding is not considered
        with torch.autograd.set_detect_anomaly(True):            
            print("Training for", epochs, "epochs.")
            
            best_avg_loss = np.inf
            best_loss_noupdates = 0
            
            np.set_printoptions(suppress=True)
            zeros = torch.zeros((batch_size, 1)).to(device)
            conf_cutoff = 0.6
            for epoch in range(1, epochs+1):
                training_loss = [] # keep track of loss per epoch                
                
                # for each conversation
                self.motion_model.train()
                for i in range(len(self.train_x_motion)):
                    conversation = self.train_x_motion[i]
                    for j in range(0, len(conversation), batch_size): # iterate through conversation in batches  
                        _batch_size = min(len(conversation)-j, batch_size)
                        h = self.motion_model.init_hidden(_batch_size) # reset each batch (stateless)
                            
                        # get batch of sequences
                        input_batch = conversation[j:j+_batch_size]
                        feature_batch = torch.tensor(input_batch) 
                        
                        # TODO this is binary audio only
                        # feature_batch = feature_batch[:,:,-1].unsqueeze(-1)
                        # TODO this is audio only pitch + intensity
                        # feature_batch = feature_batch[:,:,-2:]
                        # print(feature_batch.size())
                        if audio == False:
                            feature_batch = feature_batch[:,:,:-1] # remove audio features
                            
                        # make prediction
                        gaze_pred, h = self.motion_model(feature_batch.float(), h, pred_frames) # 128,60,2
                        
                        # get true gaze angles
                        label_gaze = torch.tensor(self.train_y[i][j:j+_batch_size]).float()
                        confidences = label_gaze[:,-pred_frames:,-2].to(device)
                        label_gaze = label_gaze[:,-pred_frames:,:2].to(device)
                        
                        # compute loss
                        optimizer.zero_grad()
                        loss_mask = confidences > conf_cutoff
                        loss1 = torch.mean((gaze_pred - label_gaze)**2, dim=-1)
                        loss1 = loss_mask * loss1
                        
                        loss = torch.mean(loss1)
                        
                        # gradient step
                        loss.backward()
                        optimizer.step()
                        
                        training_loss.append(loss.item())
                        
                avg_loss = np.mean(training_loss)
                
                # dev loss
                self.motion_model.eval()
                dev_loss = []
                for i in range(len(self.dev_windows)):
                    # print(self.dev_windows[i].shape)
                    for j in range(len(self.dev_windows[i])):
                        sequence = self.dev_windows[i][j]
                        
                        # TODO change
                        # sequence = sequence[:,-2:]
                        # if audio == False:
                        #     sequence = self.dev_windows[i][j][:,:-1]
                        h = self.motion_model.init_hidden(1)
                        x_motion = torch.tensor(sequence).unsqueeze(0)
                        
                        # TODO change
                        # x_motion = x_motion.unsqueeze(-1)
                        with torch.no_grad():
                            gaze_pred, h = self.motion_model(x_motion.float(), h, pred_frames)
                            label_gaze = torch.tensor(self.dev_outputs[i][j]).float()
                            confidences = label_gaze[-pred_frames:,-2].to(device)
                            label_gaze = label_gaze[-pred_frames:,:2].to(device)
                            gaze_pred = gaze_pred.squeeze()
                            loss_mask = confidences > conf_cutoff
                            
                            loss1 = loss_mask * torch.mean((gaze_pred - label_gaze)**2, dim=-1)#+ loss2
                            loss1 = torch.mean(loss1)
                            dev_loss.append(loss1.item())
                        
                avg_dev_loss = np.mean(dev_loss)
                
                best_loss_noupdates += 1
                if avg_dev_loss < best_avg_loss:
                    best_avg_loss = avg_dev_loss
                    best_loss_noupdates = 0
                    cached_params = self.motion_model.state_dict()
                    torch.save(cached_params, "./models/" + model_name + ".pth")
                
                print("Average loss:", avg_loss, "Dev loss:", avg_dev_loss, "Epoch:", epoch, "Lowest avg loss:", best_avg_loss)
                self.SaveTrainingOutput("./models/" + model_name, [epoch, avg_loss, avg_dev_loss, best_avg_loss], append=epoch>1)
                
                # save training model too
                torch.save(self.motion_model.state_dict(), "./models/" + model_name + "_Training.pth")
                

                    
        # TODO check file names
        if len(self.training_output) > 0:
            self.motion_model.load_state_dict(cached_params)
            torch.save(self.motion_model.state_dict(), "./models/" + model_name + ".pth")
        print("Done.")
    
        

def EvaluateModel(model_name, input_dim, hidden_dim, output_dim, gru_layers, dev_x, dev_y, test_x, test_y, step=1, audio=True):
    pred_frames = step
    
    torch.cuda.empty_cache()
    print("Loading model:", model_name, "...")
    motion_model = MotionGRUNet(input_dim, hidden_dim, 
                                output_dim, gru_layers).to(device)
    
    motion_model.load_state_dict(torch.load("./models/" + model_name + ".pth", map_location=device))
    motion_model.eval()
    print("Model loaded.")
    
    # Evaluation model on dev set
    conf_cutoff = 0.6
    motion_model.eval()
    dev_loss= []
    print("dev len:", len(dev_x), "test len:", len(test_x))
    for i in range(len(dev_x)):
        for j in range(len(dev_x[i])):
            sequence = dev_x[i][j]
            
            # TODO change
            sequence = sequence[:,-2:]
            # print(sequence.shape)
            if audio == False:
                sequence = dev_x[i][j][:,:-1]
            h = motion_model.init_hidden(1)
            x_motion = torch.tensor(sequence).unsqueeze(0)
            
            # TODO change
            # x_motion = x_motion.unsqueeze(-1)
            with torch.no_grad():
                gaze_pred, h = motion_model(x_motion.float(), h, pred_frames)
                label_gaze = torch.tensor(dev_y[i][j]).float()
                confidences = label_gaze[-pred_frames:,-2].to(device)
                label_gaze = label_gaze[-pred_frames:,:2].to(device)
                gaze_pred = gaze_pred.squeeze()
                loss_mask = confidences > conf_cutoff
                loss1 = loss_mask * torch.mean((gaze_pred - label_gaze)**2, dim=-1)
                loss1 = torch.mean(loss1)
                dev_loss.append(loss1.item())
            
    avg_dev_loss = np.mean(dev_loss)
    print("Dev set loss:", avg_dev_loss)
    
    # # Evaluation using MSE on test set
    test_loss = []
    for i in range(len(test_x)):
        for j in range(len(test_x[i])):
            sequence = test_x[i][j]
            
            # TODO change
            sequence = sequence[:,-2:]
            if audio == False:
                sequence = test_x[i][j][:,:-1]
            h = motion_model.init_hidden(1)
            x_motion = torch.tensor(sequence).unsqueeze(0)
            
            # TODO change
            # x_motion = x_motion.unsqueeze(-1)
            with torch.no_grad():
                gaze_pred, h = motion_model(x_motion.float(), h, pred_frames)
                label_gaze = torch.tensor(test_y[i][j]).float()
                confidences = label_gaze[-pred_frames:,-2].to(device)
                label_gaze = label_gaze[-pred_frames:,:2].to(device)
                gaze_pred = gaze_pred.squeeze()
                loss_mask = confidences > conf_cutoff
                loss1 = loss_mask * torch.mean((gaze_pred - label_gaze)**2, dim=-1)
                loss1 = torch.mean(loss1)
                test_loss.append(loss1.item())
            
    avg_test_loss = np.mean(test_loss)
    print("Test set loss:", avg_test_loss)
    

if __name__ == "__main__":
    lookback = training_params['lookback'] # input sequence length (seconds)
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate']
    epochs = training_params['epochs']
    seed = training_params['seed']
    step = training_params['step']
    framerate = data_params['framerate']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # load motion data features
    data_loader = FeatureLoader()
    training_paths = motion_data['training']
    
    train_windows, dev_windows, test_windows, train_outputs, dev_outputs, test_outputs, means, sds = \
    data_loader.LoadTrainingData(training_paths, framerate, lookback, which_features="all", shuffle=True, seed=seed, pred_stride=step)
    
    # init gaze mode class and split data to training/testing sets
    # Dummy variable
    audio_windows = []
    model_name = "Model_DispAudioBinary_4FClayer_LossMask_60percentTrainAugment" + str(step) + "_" + str(seed)
    gaze_model = EyeMotionModel(train_windows, audio_windows, dev_windows, test_windows, train_outputs, dev_outputs, test_outputs, framerate, audio=True)
    gaze_model.DivideData(seed)
    
    t0 = time.time()
    
    gaze_model.TrainMotionRNN(lr=learning_rate, epochs=epochs, step=step, batch_size=batch_size, model_name=model_name, audio=True, pred_frames=step)
    print("Time spent training:", str(round(time.time() - t0, 3)) + 's')
    
    input_dim = 2
    hidden_dim = 128
    output_dim = 2
    gru_layers = 3
    # EvaluateModel(model_name, input_dim, hidden_dim, output_dim, gru_layers, \
    #               dev_windows, dev_outputs, test_windows, test_outputs, step=step, audio=True)