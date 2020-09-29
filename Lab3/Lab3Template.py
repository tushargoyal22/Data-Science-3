import numpy as np
import pandas as pd
def read_data(path_to_file):
	""" Returns Pandas dataframe for given csv file
	
		Parameters
		----------
		path_to_file: string
			Given csv file
		
		Returns
		-------
		pandas.Dataframe
	"""
	pass 
def show_box_plot(attribute_name,dataframe):
	""" Displays boxplot for atrribute

		Parameters
		----------
		attribute_name: string
			Attribute selected
		dataframe: pandas.Dataframe
			DataFrame for the given dataset
		Returns
		-------
		None
	"""
	pass
def replace_outliers(dataframe):
	""" Replaces the outliers in the given dataframe
	
		Parameters
		----------
		dataframe: pandas.Dataframe
			DataFrame for the given dataset
		Returns
		-------
		pandas.Dataframe
	"""
	pass
def range(dataframe,attribute_name):
	""" Gives Range of Selected Attribute
	
		Parameters
		----------
		attribute_name: string
			Attribute selected
		dataframe: pandas.Dataframe
			DataFrame for the given dataset
		Returns
		-------
		pair(float,float)
	"""
	pass
def min_max_normalization(dataframe,range=None):
	""" Returns normalized pandas dataframe
	
		Parameters
		----------
		dataframe: pandas.Dataframe
			Dataframe for the given dataset
		range: pair(float,float) 
			Normalize between range
		Returns
		-------
		pandas.Dataframe
	"""
	pass
def standardize(dataframe):
	""" Returns standardized pandas dataframe
	
		Parameters
		----------
		dataframe: pandas.Dataframe
			Dataframe for the given dataset
		Returns
		-------
		pandas.Dataframe
	"""
	pass

def main():
	""" Main Function
		Parameters
		----------
		
		Returns
		-------
		None
	"""
	path_to_file="Your_File_Path"
	dataframe=read_data(path_to_file)
	return

if __name__=="__main__":
	main()