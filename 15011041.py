import pickle
import os

import keras
import skimage.io as io
from keras import Model
from keras.engine.saving import model_from_json
from skimage.transform import resize
import random

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D
import numpy as np


class Data:
    def __init__(self, name: str, train: list, test: list):
        self.name = name
        self.train = train
        self.test = test

    def getTrainNpArray(self):
        trainList = []
        for image in self.train:
            trainList.append(image.matrix)
        return trainList

    def getTestNpArray(self):
        testList = []
        for image in self.test:
            testList.append(image.matrix)
        return testList


class Image:
    def __init__(self, name: str, imageClass: int, matrix):
        self.name = name
        self.matrix = matrix
        self.imageClass = imageClass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.matrix


class Dist:
    def __init__(self, name, dist):
        self.name = name
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist

    def __str__(self):
        return self.dist

    def __repr__(self):
        return self.dist


def readDataFolder():
    root = "data/"
    folderNames = os.listdir(root)
    imageClass = 0
    print(folderNames)
    saveObjectToFile(folderNames, "classNames.pkl")
    for folder in folderNames:
        print(imageClass)
        data = readImageData(root, folder + "/", imageClass)
        saveObjectToFile(data, folder + ".pkl")
        del data
        imageClass += 1


def readImageData(root, building, imageClass):
    print("READING............................................. ", building)
    path = root + building
    images = []
    imageNames = os.listdir(path)
    print("DATA SIZE :\t", len(imageNames))
    for i in range(0, len(imageNames)):
        imageData = io.imread(path + imageNames[i])
        imageData = resize(imageData, (200, 200, 3))
        images.append(Image(imageNames[i], imageClass, imageData))

    random.shuffle(images)

    trainCount = (66 * len(imageNames)) // 100  # % 80 Train

    trainData = images[0:trainCount]
    testData = images[trainCount + 1:]
    del images

    print("TRAIN SET :\t", len(trainData))
    print("TEST SET  :\t", len(testData))
    print(trainData[0].name)

    return Data(building[:-1], trainData, testData)


def cnnModel(imageData, imageLabel, classSize):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(200, 200, 3), name='input_layer'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(classSize, activation='softmax', name='output_layer'))
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(imageData, imageLabel, batch_size=64, epochs=8, validation_split=0.2)

    saveModelToJSON(model)


def readObjectFromFile(filename):
    try:
        with open(filename, 'rb') as inputs:
            data = pickle.load(inputs)
            print(filename[:-4], " dosyadan okuma başarılı.")
    except FileNotFoundError:
        data = []
    return data


def saveObjectToFile(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def nearestImageWithVector():
    predictionData = np.load("prediction.npy")
    predictionTest = np.load("predictionTest.npy")
    nearestImages = []

    for i in range(0, len(predictionTest)):
        nearestDists = []
        for j in range(0, len(predictionData)):
            dist = np.linalg.norm(predictionTest[i] - predictionData[j])
            dist = Dist(j, dist)
            nearestDists.append(dist)
        nearestDists.sort()
        nearestDists = nearestDists[:5]
        nearestImages.append(nearestDists)
    print(nearestImages)


def readAllPickleFiles(classNames):
    allData = []
    if len(classNames) == 0:
        print("Pickle yok.")
        readDataFolder()
        print("Tüm fotoğraflar okundu ve pickle olarak kaydedildi.")
    else:
        i = 0
        for name in classNames:
            allData.append(readObjectFromFile(name + ".pkl"))
            print(len(allData[i].train))
            print(len(allData[i].test))
            i += 1
        print("Tüm veri pickle üzerinden okundu.")
    return allData


def convertDataToNpArrayAndSave(allData):
    imageData = []
    imageLabel = []
    for data in allData:
        imageData += data.getTestNpArray()
        for image in data.test:
            imageLabel.append(image.imageClass)

    imageData = np.array(imageData).reshape(-1, 200, 200, 3)
    np.save("imageDataTest.npy", imageData)
    imageLabel = np.array(imageLabel)
    np.save("imageLabelTest.npy", imageLabel)
    print("KAYIT BAŞARILI")
    del imageData
    del imageLabel
    imageData = []
    imageLabel = []

    for data in allData:
        imageData += data.getTrainNpArray()
        for image in data.train:
            imageLabel.append(image.imageClass)

    imageData = np.array(imageData).reshape(-1, 200, 200, 3)
    np.save("imageData.npy", imageData)
    imageLabel = np.array(imageLabel)
    np.save("imageLabel.npy", imageLabel)
    print("KAYIT BAŞARILI")


def readNpArray(dataPath, labelPath):
    imageData = np.load(dataPath)
    imageLabel = np.load(labelPath)
    print(dataPath, " ve ", labelPath + " okundu.")
    return imageData, imageLabel


def saveModelToJSON(model):
    model_json = model.to_json()
    with open("./model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./model.h5")
    print("Model kaydedildi.")


def readModelFromJSON():
    json_file = open('./model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("./model.h5")
    model.summary()
    print("Model okundu.")
    return model


def printAccuracyAndLoss(model, testData, testLabel):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(testData, testLabel, verbose=0)
    print("LOSS : %.2f" % (score[0] * 100))
    print("ACCURACY : %.2f" % (score[1] * 100))


def createAndSaveOutputLayerVectors(model, trainData, testData):
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer('output_layer').output)
    intermediate_prediction = intermediate_model.predict(trainData)
    print(intermediate_prediction.shape)
    np.save("prediction.npy", intermediate_prediction)
    intermediate_prediction = intermediate_model.predict(testData)
    print(intermediate_prediction.shape)
    np.save("predictionTest.npy", intermediate_prediction)
    print("Kaydedildi.")


def createAndSaveVGGVectors(trainData, testData, classSize):
    vggModel: Model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(200, 200, 3),
                                                     pooling='max')
    vggModel.summary()
    vggModel = Model(inputs=vggModel.input, outputs=vggModel.get_layer("fc1").output)
    vggModel.summary()
    intermediate_prediction = vggModel.predict(trainData)
    print(intermediate_prediction.shape)
    np.save("predictionVGG.npy", intermediate_prediction)
    intermediate_prediction = vggModel.predict(testData)
    print(intermediate_prediction.shape)
    np.save("predictionTestVGG.npy", intermediate_prediction)
    print("Kaydedildi.")


def main():
    # readDataFolder()
    # classNames = readObjectFromFile("classNames.pkl")
    # allData = readAllPickleFiles(classNames)
    # convertDataToNpArrayAndSave(allData)
    # trainData, trainLabel = readNpArray("imageData.npy", "imageData.npy")
    # testData, testLabel = readNpArray("imageDataTest.npy", "imageLabelTest.npy")
    # cnnModel(trainData, trainLabel, len(classNames))
    # model = readModelFromJSON()
    # printAccuracyAndLoss(model, testData, testLabel)
    # createAndSaveOutputLayerVectors(model, trainData, testData)
    # nearestImageWithVector()
    # createAndSaveVGGVectors(trainData, testData, len(classNames))
    return


if __name__ == '__main__':
    main()
