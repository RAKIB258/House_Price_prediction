from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_house_attributes(inputPath):
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	# df.head()
	# print(df)
	# load_house_attributes("C:/Users/Shawon/Desktop/uylab/project/Houses Dataset/HousesInfo.txt")
	zipcode = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().tolist()

	for(zipcode, count) in zip(zipcode,counts):
		if count < 25:
			id = df[df["zipcode"] == zipcode].index
			df.drop(id, inplace=True)
	return df

def process_house_attributes(df, train, test):
	continous = ["bedrooms", "bathrooms", "area"]
	cs = MinMaxScaler()
	traindata = cs.fit_transform(train[continous])
	testdata = cs.transform(test[continous])

	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	trainCategorical = zipBinarizer.transform(train["zipcode"])
	testCategorical = zipBinarizer.transform(test["zipcode"])

	trainX = np.hstack([trainCategorical, traindata])
	testX = np.hstack([testCategorical, testdata])

	return (trainX, testX)
	


