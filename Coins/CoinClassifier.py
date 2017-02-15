#!/usr/bin/python
from keras.models import model_from_json
from keras.optimizers import RMSprop

class CoinClassifier:
    def __init__(self, TypeClasification, OneVsTwoClasification, GoldenClassification, CopperClassification):    
        self.ModelName_TypeClasification= TypeClasification
        self.ModelName_OneVsTwoClasification= OneVsTwoClasification
        self.ModelName_GoldenClasification= GoldenClassification
        self.ModelName_CopperClasification= CopperClassification

        self.Model_TypeClasification= self.loadModel(self.ModelName_TypeClasification)
        self.Model_GoldenClassTypeClasification= self.loadModel(self.ModelName_GoldenClassification)
        self.Model_CopperClassTypeClasification= self.loadModel(self.ModelName_CopperClassification)
        
        
    def loadModel(self, modelName):
        # load json and create model
        json_file = open(modelName+".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(modelName+".h5")
        print("Loaded model "+modelName+" from disk \n")
        loaded_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy']) ## This needs to be adapted to each case...
        return loaded_model


    def getMaxScore(self, predictions):
        nlabels = predictions.shape[1]
        bestScore = -1
        index = -1
        for k in range(0, nlabels):
            if predictions[0][k] > bestScore:
                bestScore = predictions[0][k]
                index = k
        return bestScore, k

        
    def GetCoinFromLabel(CoinType, CoinLabel):
        'Converts between labels type into a sentence with the coin name'
        label = ""
        if CoinType == 0:
            if CoinLabel == 0:
                label = "1 euro"
            if CoinLabel == 1:
                label = "2 euro"
        if CoinType == 1:
            if CoinLabel == 0:
                label = "10 cent"
            if CoinLabel == 1:
                label = "20 cent"
            if CoinLabel == 2:
                label = "50 cent"
        if CoinType == 2:
            if CoinLabel == 0:
                label = "1 cent"
            if CoinLabel == 1:
                label = "2 cent"
            if CoinLabel == 2:
                label = "5 cent"
        return label
    
    def predictLabel(self, img):
        'Predicts the labels from a given image'
        #Expected input is required to have 4 dimensions
        predictedTypeLabel = self.Model_TypeClasification.predict(img)
        ScoreCoinType, CoinType = self.getMaxScore(predictedTypeLabel)

        if CoinType == 0:
            #This is a 2 or 1 euro Coin
            predictedCoinLabel = self.Model_OneVsTwoClasification.predict(img)
        if CoinType == 1:
            #This is a 50, 20, 10 cent coin
            predictedCoinLabel = self.Model_GoldenClasification.predict(img)
        if CoinType == 0:
            #This is a 2 or 1 euro Coin
            predictedCoinLabel = self.Model_CopperClasification.predict(img)

        ScoreCoinLabel, CoinLabel = self.getMaxScore(predictedCoinLabel)

        GetCoinFromLabel(CoinType, CoinLabel)
        
        return ScoreCoinType, CoinType, ScoreCoinLabel, CoinLabel

        
            
            



    
