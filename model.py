

class Xgboost:
    def __init__(self,model):
        self.model = model
    
    def classify(self, input_data):
        result = self.model.predict(input_data)
    
        if result == 0:
            result = "Negative"
        elif result == 1:
            result = "Positive"

        return result
