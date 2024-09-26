# -*- coding: utf-8 -*-
"""
Configuration file for model training
"""

# Training settings
training_params = {
    'learning_rate': 3e-5,
    'epochs': 1000,
    'batch_size': 128,
    'step': 12,
    'model_name': "GazeModel",
    'lookback': 4,
    'seed': 837
}

# Data preprocessing settings
data_params = {
    'framerate': 60,
    
    # Feature means and standard deviations for each model variation (features used)
    'Model_AudioPitchIntensity_Means': [5.788530914954428e-17,-0.0480733151208624,5.559887339510393e-17,0.08194701475311003,1.2029948225733831e-05,4.143646880764377e-07,-0.0006118539831994607],
    'Model_AudioPitchIntensity_SDs': [0.0925970668149745,0.10694791488148453,0.0735283404244664,0.113524876519268,3.8534173318933025e-05,0.0038011029156385647,824.215850609232],
    
    'Model_AudioBinary_Means': [5.788530914954428e-17,-0.0480733151208624,5.559887339510393e-17,0.08194701475311003,1.2029948225733831e-05,0.5481979915379518],
    'Model_AudioPitchBinary_SDs': [0.0925970668149745,0.10694791488148453,0.0735283404244664,0.113524876519268,3.8534173318933025e-05,0.4976715318477716],
    
    'Model_NoAudio_Means': [5.788530914954428e-17,-0.0480733151208624,5.559887339510393e-17,0.08194701475311003,1.2029948225733831e-05],
    'Model_NoAudio_SDs': [0.0925970668149745,0.10694791488148453,0.0735283404244664,0.113524876519268,3.8534173318933025e-05],
}


# motion data to use for training
motion_data = {
    'training' : ["./TrainingData/2021_12_01_01_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_03_01_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_03_02_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_03_03_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_03_05_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_07_01_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_07_02_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_08_01_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_08_02_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_08_03_Smoothed_S5_Lerp1_Transformed.csv",
                  "./TrainingData/2021_12_08_04_Smoothed_S5_Lerp1_Transformed.csv",
                  
                   "./TrainingData/2021_12_01_01_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_03_01_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_03_02_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_03_03_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_03_05_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_07_01_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_07_02_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_08_01_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_08_02_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_08_03_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv",
                   "./TrainingData/2021_12_08_04_Smoothed_S5_Lerp1_Transformed_mirroredTheta.csv"
                  ]
}