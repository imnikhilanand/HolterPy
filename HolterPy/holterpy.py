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
        self.data_np_df = self.csv_file_pd_df.iloc[0:,0:2].values

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
    def plot_R_peaks(self):
        self.get_R_np_arr = np.array(self.final_arr_R_values)
        plt.plot(self.data_np_df[0:,0])
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
    def plot_P_peaks(self):
        self.get_P_np_arr = np.array(self.final_arr_P_values)
        plt.plot(self.data_np_df[0:,0])
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
    
    def plot_T_peaks(self):
        self.get_T_np_arr = np.array(self.final_arr_T_values)
        plt.plot(self.data_np_df[0:,0])
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

#running objects to verify if its working or not        
"""
obj = holterpy()
obj.load_csv('noraml_ecg.csv',freq=500,header=None)
a = obj.return_pd_df()
d = obj.return_np_df()
c = obj.get_R_val()
obj.plot_R_peaks()
b = obj.get_P_val()
obj.plot_P_peaks()
t = obj.get_T_val()
"""








""" Miscallaneous task 

c_np = np.array(c)

#lets plot r-peaks
plt.plot(b[0:1000,0])
plt.scatter(c_np[0:4,0],c_np[0:4,1],color='red')
plt.show()


obj.plot_pd_df(9000,12000)

#additional working process        
k=pd.read_csv('noraml_ecg.csv',header=None)        
k[0:1000].plot()


curr = []
final_arr = []
flag=0

for i in range(0,1000):
    if (float(d[i][0]) > float(d[i][1])):
        curr.append([i,d[i][0],d[i][1]])
        flag=1
    elif (float(d[i][0]) < float(d[i][1])) and flag == 1:
        max_num_index = curr[0][0]
        max_num = curr[0][1]
        for i in range(0,len(curr)):
            if curr[i][1] > max_num:
                max_num_index = curr[i][0]
                max_num = curr[i][1]
        flag=0                  
        final_arr.append([max_num_index,max_num])
        #curr = []
 
#final_arr

#here we are taking the number of records per second to be 500
#the limit would be 60 to 100 data points befor the R peak
p_values_arr = []
for i in range(0,len(final_arr)):
    r_x = final_arr[i][0]
    r_y = final_arr[i][1]
    if r_x > 100:
        max_p_dyn = b[r_x-100][0]
        for j in range(r_x-100,r_x-60):
            if b[j][0] > max_p_dyn :
                max_p_dyn = b[j][0]
                max_p_x_c = j
                
        p_values_arr.append([max_p_x_c,max_p_dyn])    
        
        
        
        
        
p_val_num = np.array(p_values_arr)        
r_final_num = np.array(final_arr)     
       


#t-value

t_values_arr = []
for i in range(0,len(final_arr)):
    r_x = final_arr[i][0]
    r_y = final_arr[i][1]
    if r_x + 200 < len(d):
        max_t_dyn = d[r_x+100][0]
        for j in range(r_x+100,r_x+200):
            if d[j][0] > max_t_dyn :
                max_t_dyn = d[j][0]
                max_t_x_c = j
                
        t_values_arr.append([max_t_x_c,max_t_dyn])    

t_val_num = np.array(t_values_arr)        
r_final_num = np.array(final_arr)     



plt.plot(d[0:1000,0])   
plt.scatter(r_final_num[0:,0],r_final_num[0:,1],color='red')        
plt.scatter(p_val_num[0:,0],p_val_num[0:,1],color='green')        
plt.scatter(t_val_num[0:,0],t_val_num[0:,1],color='yellow')        
plt.show()

"""


        