"""This entire Library is desined for the analysis of data received from Holter Device.
Every parameter that needs to be find out for the analysis of ECG are carried out here.
"""

"""Importing dependent libraries
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

"""HolterPy class begins here
"""

class holterpy:
    
    #Passing the constructor 
    def __init__(self):
        pass
    
    #overloading the load_csv function for header in the column
    def load_csv(self,csv_file,freq,header=None):
        if header is None:
            self.csv_file_pd_df = pd.read_csv(csv_file,header=None,names=['data'])    
        elif header is not None:
            self.csv_file_pd_df = pd.read_csv(csv_file)
            self.csv_file_pd_df.rename(columns = {list(self.csv_file_pd_df)[0]:'data'})    
   
        self.mov_avg = self.csv_file_pd_df['data'].rolling(int(0.75*freq)).mean()
        self.avg_data = (np.mean(self.csv_file_pd_df.data))
        self.mov_avg = [self.avg_data if math.isnan(x) else x for x in self.mov_avg]
        self.mov_avg = [x*1.2 for x in self.mov_avg]
        self.csv_file_pd_df['rolling_mean'] = self.mov_avg
        self.temp_series = np.arange(0,len(self.csv_file_pd_df)-1,1,np.int)
        self.pd_series_ind = pd.Series(self.temp_series)
        self.csv_file_pd_df['index'] = self.pd_series_ind
        self.data_np_df = self.csv_file_pd_df.iloc[0:,0:3].values

    #to get r-peaks in the ECG data
    def get_R_val(self):
        self.curr = []
        self.final_arr_R_values = []
        self.flag=0
        
        for i in range(0,len(self.data_np_df)):
            if (float(self.data_np_df[i][0]) > float(self.data_np_df[i][1])):
                self.curr.append([i,self.data_np_df[i][0],self.data_np_df[i][1]])
                self.flag=1
            elif (float(self.data_np_df[i][0]) < float(self.data_np_df[i][1])) and self.flag == 1:
                self.max_num_index = self.curr[0][0]
                self.max_num = self.curr[0][1]
                for i in range(0,len(self.curr)):
                    if self.curr[i][1] > self.max_num:
                        self.max_num_index = self.curr[i][0]
                        self.max_num = self.curr[i][1]
                self.flag=0                  
                self.final_arr_R_values.append([self.max_num_index,self.max_num])
                self.curr = []
        
        return self.final_arr_R_values            
        
    #to get R-peaks on the plot
    def plot_R_peaks(self, final_arr_R_values, lower_bound=0, upper_bound=0):
        self.temp_array_bw_bounds = []
        for i in range(0,len(final_arr_R_values)):
            if final_arr_R_values[i][0] < upper_bound and final_arr_R_values[i][0] > lower_bound:
                self.temp_array_bw_bounds.append([final_arr_R_values[i][0],final_arr_R_values[i][1]])
            
        self.get_R_np_arr = np.array(self.temp_array_bw_bounds)
        plt.plot(self.data_np_df[lower_bound:upper_bound,2],self.data_np_df[lower_bound:upper_bound,0])
        plt.scatter(self.get_R_np_arr[0:,0],self.get_R_np_arr[0:,1],color='red')    
        plt.show()
        
    #to get P-peaks on the plot
    def get_P_val(self):
        self.final_arr_P_values = []
        for i in range(0,len(self.final_arr_R_values)):
            self.x_cor_R = self.final_arr_R_values[i][0]
            self.y_cor_R = self.final_arr_R_values[i][1]
            if self.x_cor_R > 100:
                self.max_p_dyn = self.data_np_df[self.x_cor_R-100][0]
                for j in range(self.x_cor_R-100,self.x_cor_R-60):
                    if self.data_np_df[j][0] > self.max_p_dyn :
                        self.max_p_dyn = self.data_np_df[j][0]
                        self.max_p_x_c = j
                self.final_arr_P_values.append([self.max_p_x_c,self.max_p_dyn])    
        
        return self.final_arr_P_values

    #to get P-peaks on the plot
    def plot_P_peaks(self,final_arr_P_values,lower_bound=0, upper_bound=0):
        self.temp_array_bw_bounds = []
        for i in range(0,len(final_arr_P_values)):
            if final_arr_P_values[i][0] < upper_bound and final_arr_P_values[i][0] > lower_bound:
                self.temp_array_bw_bounds.append([final_arr_P_values[i][0],final_arr_P_values[i][1]])
                
        self.get_P_np_arr = np.array(self.temp_array_bw_bounds)
        plt.plot(self.data_np_df[lower_bound:upper_bound,2],self.data_np_df[lower_bound:upper_bound,0])
        plt.scatter(self.get_P_np_arr[0:,0],self.get_P_np_arr[0:,1],color='green')    
        plt.show()

    #to get T-peaks on the plot
    def get_T_val(self):
        self.final_arr_T_values = []
        for i in range(0,len(self.final_arr_R_values)):
            self.x_cor_R = self.final_arr_R_values[i][0]
            self.y_cor_R = self.final_arr_R_values[i][1]
            if self.x_cor_R + 200 < len(self.data_np_df):
                self.max_t_dyn = self.data_np_df[self.x_cor_R + 100][0]
                for j in range(self.x_cor_R + 100,self.x_cor_R + 200):
                    if self.data_np_df[j][0] > self.max_t_dyn :
                        self.max_t_dyn = self.data_np_df[j][0]
                        self.max_t_x_c = j
                self.final_arr_T_values.append([self.max_t_x_c,self.max_t_dyn])    
       
        return self.final_arr_T_values
    
    def plot_T_peaks(self,final_arr_T_values,lower_bound=0, upper_bound=0):
        self.temp_array_bw_bounds = []
        for i in range(0,len(final_arr_T_values)):
            if final_arr_T_values[i][0] < upper_bound and final_arr_T_values[i][0] > lower_bound:
                self.temp_array_bw_bounds.append([final_arr_T_values[i][0],final_arr_T_values[i][1]])
                
        self.get_T_np_arr = np.array(self.temp_array_bw_bounds)
        plt.plot(self.data_np_df[lower_bound:upper_bound,2],self.data_np_df[lower_bound:upper_bound,0])
        plt.scatter(self.get_T_np_arr[0:,0],self.get_T_np_arr[0:,1],color='yellow')    
        plt.show()
    
    def return_np_df(self):
        return self.data_np_df            
    
    #returning the pandas dataframe        
    def return_pd_df(self):
        return self.csv_file_pd_df
    
    #plotting the pandas dataframe
    def plot_pd_df(self,a,b):   
        plt.plot(self.csv_file_pd_df['data'][a:b])
        plt.plot(self.csv_file_pd_df['rolling_mean'][a:b],color='orange')
        plt.show()
        
"""HolterPy class ends here
"""



  
