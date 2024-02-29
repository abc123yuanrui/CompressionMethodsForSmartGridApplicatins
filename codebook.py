from numba import jit
from sklearn.neighbors import DistanceMetric
import scipy
import pandas as pd
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
import numpy as np
import numba

class Code_book:
  '''
  This class includes different methods and implementations of our proposed codebook methods with Similarity Profile
  '''
  def __init__(self, time_series, m, mode='euclidean'):
    self.time_series = time_series
    self.m = m
    self.pattern_stack = []
    self.mode = mode
    self.sliding = time_series.shape[0]//m
    self.filtered_demand = time_series

  def weighted_DTW(self, series_1, series_2, weight):
    #### Series_1 and Series_2 are time series list, weight is weighted matrix list with two dimention
    #### default weight can be np.ones(len(series_1))
    #### weight cannot be negative (can make it square but need to think)
    l1 = len(series_1)
    l2 = len(series_2)
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.
    for i in range(l1):
        for j in range(l2):
            diff = (series_1[i] - series_2[j])
            distance = diff*diff*weight[i][j]
            cum_sum[i + 1, j + 1] = distance
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                            cum_sum[i + 1, j],
                                            cum_sum[i, j])
    acc_cost_mat = cum_sum[1:, 1:]
    return np.sqrt(acc_cost_mat[-1][-1])
  
  def flex_distance(self, series_a, series_b, weight_amp_metrix, weight_tem_metrix):
    '''by default, the weighted_series is the same as series_a and sereis_b and in a numpy array format'''
    ### weight_matrix: the weight matrix for the distance calculation based on residual distribution (how to convert)
    sliding = series_a.shape[0]
    # build the distance matrix
    amplitude_matrix = np.zeros((sliding, sliding))
    temporal_matrix = np.zeros((sliding, sliding))
    for i in range(sliding):
        for j in range(sliding):
            amplitude_matrix[i][j] = series_a[i]-series_b[j]
            temporal_matrix[i][j] = i-j
    # build the merged matrix
    merged_matrix = np.abs(amplitude_matrix)*weight_amp_metrix + np.abs(temporal_matrix)*weight_tem_metrix
    # build the distance
    row_ind, col_ind = linear_sum_assignment(merged_matrix)
    # return merged_matrix[row_ind, col_ind].mean(), row_ind, col_ind
    return merged_matrix[row_ind, col_ind].mean()
  
  def get_distance(self, series_1, series_2):
    if self.mode == 'euclidean':
      dis = DistanceMetric.get_metric('euclidean')
      distance = dis.pairwise([series_1, series_2])[0][1]
    elif self.mode == 'DTW':
      weighted_matrix = np.ones((self.m, self.m))
      distance = self.weighted_DTW(series_1, series_2, weighted_matrix)
    elif self.mode == 'wasserstein':
      distance = scipy.stats.wasserstein_distance(series_1, series_2)
    elif self.mode == 'flexibilityD':
      distance = self.flex_distance(series_1, series_2, np.ones((self.m))*self.m, np.ones((self.m)))       
    return distance

  # def get_distance_matrix(self, series_data):
  #   '''
  #   This function is to get the distance matrix of the time series by different methods
  #   need to change
  #   '''
  #   m=self.m
  #   sliding = series_data.shape[0]//m
  #   distance_matrix = np.ones((sliding, sliding))
  #   if self.mode == 'euclidean':
  #     dis = DistanceMetric.get_metric('euclidean')
  #     ed_matrix = np.zeros((sliding, m))
  #     for j in range(sliding):
  #       ed_matrix[j] = series_data[j*m:j*m+m]
  #     distance_matrix = dis.pairwise(ed_matrix)
  #   elif self.mode in ['DTW', 'flexibilityD']:
  #     weighted_matrix = np.ones((m, m))
  #     for i in range(sliding):
  #       pattern = series_data[i*m:i*m+m]
  #       for j in range(sliding):
  #           # distance_matrix is inefficient, might needs to be changed 
  #           distance = self.weighted_DTW(pattern, series_data[j*m:j*m+m], weighted_matrix) if self.mode == 'DTW' else self.flex_distance(pattern, series_data[j*m:j*m+m], np.ones((self.m))*self.m, np.ones((self.m)))
  #           distance_matrix[i][j] = distance
  #   elif self.mode == 'wasserstein':
  #     for i in range(sliding):
  #       pattern = series_data[i*m:i*m+m]
  #       for j in range(sliding):
  #           distance = scipy.stats.wasserstein_distance(pattern, series_data[j*m:j*m+m])
  #           distance_matrix[i][j] = distance
  #   return distance_matrix
  def get_distance_matrix(self):
    m=self.m
    days = self.time_series.shape[0]//m
    distance_matrix = np.full((days, days), np.inf)
    for i in range(days):
        current_pattern = self.time_series[i*m:i*m+m]
        for j in range(days):
            distance = self.get_distance(current_pattern, self.time_series[j*m:j*m+m])
            distance_matrix[i][j] = distance

    # Compute the mean of all elements in the matrix
    mean_value = np.mean(distance_matrix)

    # Compute quantile values (e.g., 25th, 50th, and 75th percentiles) for all elements in the matrix
    quantiles = np.quantile(distance_matrix, [0.25, 0.50, 0.75])
    print(f"Mean: {mean_value}, 25th Percentile: {quantiles[0]}, 50th Percentile: {quantiles[1]}, 75th Percentile: {quantiles[2]}")
    self.distance_matrix = distance_matrix
    return distance_matrix
  
  def simple_decomposition(self, series_data):
    ''' threshold 0,0 means the automatic mode'''
    m=self.m
    sliding = series_data.shape[0]//m
    distance_matrix = self.get_distance_matrix(series_data)
    distance_profile = np.zeros((sliding))
    discord_profile = np.zeros((sliding))
    average_profile = np.zeros((sliding))
    filtered_series = np.array([])
    distance_stack = np.array([])
    labels = ['normal' for i in range(sliding)]
    for i in range(sliding):
            for j in range(sliding):
                # distance_matrix is inefficient, might needs to be changed 
                if i!=j:
                  distance = distance_matrix[i][j]
                  distance_stack = np.append(distance_stack, [distance], axis=0)
            average_profile[i] = np.average(distance_matrix[i,:])
    d_max = np.sort(distance_stack)[-1]
    # if thres_m == 0 and thres_d == 0:
  #       thres_m = np.median(distance_stack)
  #       thres_d = np.median(distance_stack)
  ## Split it into two parts by n
    pos = distance_stack.shape[0]//4
    thres_m = np.sort(distance_stack)[pos]
    print(thres_m)
    thres_d = np.sort(distance_stack)[-pos]
    print(thres_d)
    for k in range(sliding):  
            counter_m = 0
            counter_d = 0
            for l in range(sliding):
                distance = distance_matrix[k][l]
                if distance < thres_m:
                    counter_m+=1
                if distance > thres_d:
                    counter_d+=1
            distance_profile[k] = counter_m - average_profile[k]/d_max if d_max else counter_m
            discord_profile[k] = counter_d + average_profile[k]/d_max if d_max else counter_m
    RM = np.argsort(distance_profile)[-1]
    DS = np.argsort(discord_profile)[-1]
    for i in range(sliding):
        distance_m = distance_matrix[RM][i]
        distance_d = distance_matrix[DS][i]
        if distance_m<thres_m:
          labels[i] = 'repeated patterns'
        elif distance_d<=thres_m:
          labels[i] = 'abnormals'
        else:
          filtered_series = np.append(filtered_series, series_data[i*m:i*m+m], axis=0)
    labels[RM] = 'RM'
    labels[DS] = 'Discord'
    return distance_matrix, labels, filtered_series

  def desolve_time_series(self):
    m = self.m
    days = self.filtered_demand.shape[0]//m
    sample_Demand = self.filtered_demand
    counter = 0
    if days>4:
        test_matrix, labels, self.filtered_demand = self.simple_decomposition(sample_Demand)
        self.pattern_stack, normal_labels = self.desolve_time_series()
  #       print(normal_labels)
        for i in range(days):
          if labels[i] == 'RM':
            RM_name = 'RM-'+str(len(self.pattern_stack))
            labels[i] = RM_name
            RM_pattern = sample_Demand[i*m:i*m+m]
            self.pattern_stack.append(RM_pattern)
          if labels[i] == 'Discord':
            Discord_name = 'Discord-'+str(len(self.pattern_stack))
            labels[i] = Discord_name
            Discord_pattern = sample_Demand[i*m:i*m+m]
            self.pattern_stack.append(Discord_pattern)
        for i in range(days):
          if labels[i] == 'repeated patterns':
            labels[i] = RM_name
  #           sample_Demand[i*24:i*24+24] = RM_pattern
          if labels[i] == 'abnormals':
            labels[i] = Discord_name
  #           sample_Demand[i*24:i*24+24] = Discord_pattern
          if labels[i] == 'normal':
            labels[i] = normal_labels[counter]
            counter+=1
        return self.pattern_stack, labels
    else:
        [self.pattern_stack.append(sample_Demand[i*m:i*m+m]) for i in range(days)]        
        return self.pattern_stack, ['reminders-'+str(i) for i in range(days)]

  def desolve_time_series_thre(self, threshold):
    '''Do not need to rebuild the distance matrix if it is already built'''
    m = self.m
    days = self.time_series.shape[0]//m
    labels = []
    for i in range(days):
        current_pattern = self.time_series[i*m:i*m+m]
        distance_stack = []
        if self.pattern_stack:
            for j in self.pattern_stack:
                distance = self.get_distance(j, current_pattern)
                distance_stack.append(distance)
            if min(distance_stack)<threshold:
                labels.append('patterns-'+str(distance_stack.index(min(distance_stack))))
            else:
                self.pattern_stack.append(current_pattern)
                labels.append('patterns-'+str(len(self.pattern_stack)-1))
                # print('pattern stack distance is :', min(distance_stack))
        else:
            self.pattern_stack.append(current_pattern)
            labels.append('patterns-'+str(len(self.pattern_stack)-1))
    return self.pattern_stack, labels

  def time_series_recover(self, labels):
    time_series = []
    indexs = [int(i.split('-')[1]) for i in labels]
    for i in indexs:
      time_series.extend(self.pattern_stack[i])
    return np.array(time_series)
  

## Other functions 
def scaler_normalization(series_ori, m):
  sliding = series_ori.shape[0]//m
  sample_series = series_ori.reshape(sliding, m)
  scaler_max = np.max(sample_series, axis=1)
  scaler_min = np.min(sample_series, axis=1)
  scaler_10 = (scaler_max - scaler_min)/10
  scaler_10 = scaler_10.repeat(m)
  series_normalised = series_ori/scaler_10
  series_normalised = np.nan_to_num(series_normalised, nan=0)
  return series_normalised, scaler_10

def scaler_recoverary(series_normalised, scaler_10):
  return series_normalised*scaler_10

'''
Build local_weight_regression
'''
def kernel(data, point, xmat, k):
    m,n = np.shape(xmat)
    ws = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - data[j]
        ws[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return ws
 
def local_weight(data, point, xmat, ymat, k):
    wei = kernel(data, point, xmat, k)
    return (data.T*(wei*data)).I*(data.T*(wei*ymat.T))
     
def local_weight_regression(xmat, ymat, k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*local_weight(xmat, xmat[i],xmat,ymat,k)
    return ypred