import random
import numpy as np
import pandas as pd


def sequence_generator(data,L,CH0,CH1,CH2,CH3,W,D,O,c,e,augment=None,Eaugment=None,stride=1):
  sequence_group = []
  for i in range(0, len(data)-L-D, stride):
    sequence_group.append(data.iloc[i:i+L+D])
  data_= np.array([df.values for df in sequence_group])
  input1 = data_[:,:L,CH0:CH1]
  input2 = data_[:,:L,CH2:CH3]
  input = np.concatenate((input1,input2),axis=2)
  target = data_[:,-1,W:W+O]
  #input = (input-input.mean(axis=0))/input.std(axis=0)
  #input = (input-input.mean(axis=0))/input.std(axis=0)
  if augment ==1:
      augment_data = input.copy()
      augment_data2 = input.copy()
      augment_data3 = input.copy()
      increase_percentages = np.random.uniform(low=0.1, high=0.9, size=(input.shape[0], int(c/2)))
      decrease_percentages = np.random.uniform(low=0.1, high=0.9, size=(input.shape[0], c-int(c/2)))
      #increase_percentages = 0.8 * np.random.beta(2, 5,(input.shape[0], int(c/2)) ) + 0.1
      #decrease_percentages = 0.8 * np.random.beta(2, 5,(input.shape[0], int(c/2)) ) + 0.1
      for i in range(input.shape[0]):
        np.random.shuffle(increase_percentages[i])
        np.random.shuffle(decrease_percentages[i])
      col_indices = [random.sample(range(0,c), c) for i in range(input.shape[0])]
      for i in range(input.shape[0]):
        for j, col in enumerate(col_indices[i][:int(c/2)]):
          augment_data[i, :, col] = input[i, :, col] * (1 + increase_percentages[i, j])+np.random.uniform(low=-0.1, high=0.1)
        for j, col in enumerate(col_indices[i][int(c/2):]):
          augment_data[i, :, col] = input[i, :, col] * (1 - decrease_percentages[i, j])+np.random.uniform(low=-0.1, high=0.1)
        #emg aug
        if Eaugment == 1:
          #mean_values = augment_data2[i,:,c:c+e].mean(axis=0)
          mean_values = augment_data2[i,:,0:e].mean(axis=0)
          max_channel_index = np.argmax(mean_values)
          chosen_number = random.choice([0, 1])
          crange = [8,9,10,11,12,13,14,15]
          crange_e = crange+crange[:3]
          prange = crange_e[max_channel_index:max_channel_index+4]
        #comp = np.random.uniform(low=0, high=0.3)

          #for j in range(c,augment_data2.shape[2]):
          for j in range(0,e):
            if j == max_channel_index:
              if chosen_number == 0:
                augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
              else:
                augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
            else:
              if chosen_number == 0:
                augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
              else:
                augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
          augment_data3[i, :, 0:e] = augment_data2[i, :, 0:e]
          augment_data3[i, :, e:c] = augment_data[i, :, e:c]

      if Eaugment == 1:
        input = np.concatenate((input,augment_data,augment_data2),axis=0)
        target = np.concatenate((target,target,target),axis=0)
      else:
        input = np.concatenate((input,augment_data),axis=0)
        target = np.concatenate((target,target),axis=0)

  if augment == 2:
    augment_data2 = input.copy()
    mean_values = augment_data2[i,:,:].mean(axis=0)
    max_channel_index = np.argmax(mean_values)
    chosen_number = random.choice([0, 1])
    crange = [8,9,10,11,12,13,14,15]
    crange_e = crange+crange[:3]
    prange = crange_e[max_channel_index:max_channel_index+4]
        #comp = np.random.uniform(low=0, high=0.3)

    for j in range(c,augment_data2.shape[2]):
      if j == max_channel_index:
        if chosen_number == 0:
          augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
        else:
          augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
      else:
        if chosen_number == 0:
          augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
        else:
          augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
    input = np.concatenate((input,augment_data2),axis=0)
    target = np.concatenate((target,target),axis=0)
      #del_indices = np.random.rand(input.shape[0]) < 0.5
      #input = input[del_indices]
      #target = target[del_indices]

  return input,target