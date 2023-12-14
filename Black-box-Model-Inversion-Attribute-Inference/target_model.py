from bigml.model import Model
from bigml.api import BigML
from bigml.deepnet import Deepnet
from sklearn import tree

# the dictionaries store the resource ids of the target models
target_model_dict_dt = {
    'Adult': 'model/621ff90b8f679a67b30004fe',
    'fivethirtyeight': 'model/621ffc908f679a67b70004a8',
    'GSS': 'model/621ffb3b8f679a67ac0004a8',
    'original': 'model/657ab37dbbe7434a13da49b9',
    'ctgan_1': 'model/657ada94be9d6a0ca6dfa83a'
}

target_model_dict_dnn = {
    'Adult': 'deepnet/621ff969aba2df5ee400055d',
    'fivethirtyeight': 'deepnet/621ffcdd8f679a67ac0004ab',
    'GSS': 'deepnet/621ffb578f679a67b3000503',
    'original': 'deepnet/657acfc06486b409f6a950ca'
}

class TargetModel:
    def __init__(self, params):
        self.model_type = params.target_model_type
        self.model_dataset = params.dataset
        
        if self.model_type == 'DT':
            if self.model_dataset == 'Adult' or self.model_dataset == 'fivethirtyeight' or self.model_dataset == 'GSS':
                self.model = Model(target_model_dict_dt[self.model_dataset], api=BigML("usenixmiai", "d03f694d9f2e250a5b625ebe154a4f4159f6c338", domain="bigml.io"))
            else:
                self.model = Model(target_model_dict_dt[self.model_dataset], api=BigML("netPrivacy", "76dce257fd09ee6ea9ad74a39cd6279df7193eb8", domain="bigml.io"))

        elif self.model_type == 'DNN':
            if self.model_dataset == 'Adult' or self.model_dataset == 'fivethirtyeight' or self.model_dataset == 'GSS':
                self.model = Deepnet(target_model_dict_dnn[self.model_dataset], api=BigML("usenixmiai", "d03f694d9f2e250a5b625ebe154a4f4159f6c338", domain="bigml.io"))
            else:
                self.model = Deepnet(target_model_dict_dt[self.model_dataset], api=BigML("netPrivacy", "76dce257fd09ee6ea9ad74a39cd6279df7193eb8", domain="bigml.io"))
        else:
            raise ValueError(f'Model type {self.model_type} is not part of the supported model types')
                          