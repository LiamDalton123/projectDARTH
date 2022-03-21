from VADLite.ConfigVAD import *


class Classifier:
    @staticmethod
    def Classify(features: [float]) -> bool:
        classification: bool = False
        wx = 0.0  # Sum of coefficient x features
        if ConfigVAD.shouldNormalize:
            features = Classifier.normalizeFeatures(features)

        # Implement decision function of linear SVM: y = wx +b
        for i in range(0, len(ConfigVAD.COEFFICIENTS)):
            coeff = ConfigVAD.COEFFICIENTS[i]
            feature = features[i]
            wx += (coeff * features[i])

        y = wx + ConfigVAD.INTERCEPT

        if y > 0:
            classification = True

        # Return result
        return classification

    @staticmethod
    def normalizeFeatures(features):
        normFeatures = [0.0] * (len(features) - 1)

        for i in range(0, len(features) - 1):
            normFeatures[i] = (features[i + 1] - ConfigVAD.MEAN[i]) / ConfigVAD.SCALER[i]

        return features
