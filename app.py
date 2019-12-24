from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from model import datamodels
from model import models
import numpy as np
import argparse
import locale
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

print("house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datamodels.load_house_attributes(inputPath)

print("training/testing split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42)

maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

print("[INFO] processing data...")
(trainX, testX) = datamodels.process_house_attributes(df, train, test)

model = models.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=100, batch_size=2)

print("[INFO] predicting house prices...")
preds = model.predict(testX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))