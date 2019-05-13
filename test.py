"""" Importing the HolterPy Library
"""
from HolterPy import holterpy as holterpy

#running objects to verify if its working or not        
obj = holterpy()
obj.load_csv('noraml_ecg.csv',freq=500,header=None)
pandas_dataframe = obj.return_pd_df()
numpy_dataframe = obj.return_np_df()
R_peaks = obj.get_R_val()
obj.plot_R_peaks(R_peaks, lower_bound = 1000, upper_bound = 3000)
P_peaks = obj.get_P_val()
obj.plot_P_peaks(P_peaks, lower_bound = 1000, upper_bound = 3000)
T_peaks = obj.get_T_val()
obj.plot_T_peaks(T_peaks, lower_bound = 1000, upper_bound = 3000)

