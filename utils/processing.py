from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def scaler(df, scaler_type: object, numeric_cols: list, object_cols: list, tag_cols: list):
	'''
	:param df: dataframe of features
	:type df: Pandas dataframe
	:param scaler_type: string that must be MinMax or Standard
	:type index: object
	:param numeric_cols: list of column names as strings
	:type index: list
	:param object_cols: list of column names as strings
	:type index: list
	:param tag_cols: list of column names as strings
	:type index: list
    :returns: dataframe
	This function takes a dataframe and a selected numeric scaler type and returns a dataframe
	'''

	# Create a new DataFrame for processed data
	new_df = pd.DataFrame()

	# Standard scaling of numeric features
	if scaler_type == 'MinMax':
		scaler = MinMaxScaler()
	elif scaler_type == 'Standard':
		scaler = StandardScaler()
	else:
		raise ValueError('Scaler type must be MinMax or Standard') 
	new_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

	# One-hot encoding of object features
	encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
	object_features = pd.DataFrame(encoder.fit_transform(df[object_cols]))
	object_feature_names = encoder.get_feature_names_out(object_cols)
	object_features.columns = object_feature_names
	new_df = pd.concat([new_df, object_features], axis=1)

	# One-hot encoding of tags
	tags = pd.concat([df[col].apply(pd.Series).stack().str.get_dummies().sum(level=0) for col in tag_cols], axis=1)

	return pd.concat([new_df, tags], axis=1)
