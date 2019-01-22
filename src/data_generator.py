import pandas as pd
import numpy as np
import os

def generate_dataset():
	train_data_df = pd.read_csv("./../data/training-dataset/train.csv", encoding="ISO-8859-1")
	test_data_df = pd.read_csv("./../data/testing-dataset/test.csv", encoding="ISO-8859-1")

	positive_train_data_df = train_data_df[train_data_df['Patient_Tag'] == 1]
	positive_train_data = positive_train_data_df[['TRANS_CONV_TEXT']]

	negative_train_data_df = train_data_df[train_data_df['Patient_Tag'] == 0]
	negative_train_data = negative_train_data_df[['TRANS_CONV_TEXT']]

	test_data = test_data_df[['TRANS_CONV_TEXT']]

	for index, row in positive_train_data.iterrows():
		file_handler1 = open("./../data/training-dataset/patient_conversation-positive.txt", 'a', encoding='utf-8')
		file_handler1.write(str(row[0])+"\n")
		file_handler1.close()

	for index, row in negative_train_data.iterrows():
		file_handler2 = open("./../data/training-dataset/patient_conversation-negative.txt", 'a', encoding='utf-8')
		file_handler2.write(str(row[0])+"\n")
		file_handler2.close()	

	for index, row in test_data.iterrows():
		file_handler3 = open("./../data/testing-dataset/patient_conversations-test.txt", 'a', encoding='utf-8')
		file_handler3.write(str(row[0])+"\n")
		file_handler3.close()