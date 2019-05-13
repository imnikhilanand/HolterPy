"""" Importing the HolterPy Library
"""
from HolterPy import holterpy as holterpy

#running objects to verify if its working or not        
obj = holterpy()
obj.load_csv('noraml_ecg.csv',freq=500,header=None)
pandas_dataframe = obj.return_pd_df()
numpy_dataframe = obj.return_np_df()
get_R_peaks_array = obj.get_R_val()
obj.plot_R_peaks()
get_P_peaks_array = obj.get_P_val()
obj.plot_P_peaks()
get_T_peaks_array = obj.get_T_val()
obj.plot_T_peaks()

