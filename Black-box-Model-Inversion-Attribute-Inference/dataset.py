import pandas as pd
from utils.parameters import Params
from sklearn import tree

y_labels_dict = {
    'Adult': {'marital': ['Married', 'Single']},
    'GSS': {'xmovie': ['x_yes', 'x_no']},
    'fivethirtyeight' : {'alcohol': ['Yes', 'No'], 'age': ['18-29', '30-44', '45-60', '> 60']},
    'original': {'Destination': ['GATEWAY_IP', 'CAMERA_IP', 'DOMAIN_IP']},
    'ctgan_1': {'Destination': ['GATEWAY_IP', 'CAMERA_IP', 'DOMAIN_IP']},
    'kg_ctgan': {'Destination': ['GATEWAY_IP', 'CAMERA_IP', 'DOMAIN_IP']},
    'octgan': {'Destination': ['GATEWAY_IP', 'CAMERA_IP', 'DOMAIN_IP']},
    'pategan': {'Destination': ['GATEWAY_IP', 'CAMERA_IP', 'DOMAIN_IP']},
    'tablegan': {'Destination': ['GATEWAY_IP', 'CAMERA_IP', 'DOMAIN_IP']},
    'ctgan_dp_gan': {'Destination': ['GATEWAY_IP', 'CAMERA_IP', 'DOMAIN_IP']}
}

# the dataset class
# name contains the name
# sensitie_vals contain all possible values for all sensitive attributes
# sensitive_attributes store all the sensitive attribute names
# ground truths store all the actual sensitive values for each instance before dropping the column
# missing_nonsensitive_attributes and missing_nsa_vals follow the same definition as their sensitive counterpart

class Dataset:
    def __init__(self, params):
        self.name = params.dataset
        if self.name == 'Adult':
            dataset_path = 'data/Adult_35222.csv'
            self.y_attr = 'income'
        elif self.name == 'GSS':
            dataset_path = 'data/GSS_15235.csv'
            self.y_attr = 'hapmar'
        elif self.name == 'fivethirtyeight':
            dataset_path = 'data/fte_full.csv'
            self.y_attr = 'steak_type'
        elif self.name == 'original':
            dataset_path = 'data/original.csv'
            self.y_attr = 'Attack'
        elif self.name == 'ctgan_1':
            dataset_path = 'data/ctgan_1.csv'
            self.y_attr = 'Attack'
        elif self.name == 'kg_ctgan':
            dataset_path = 'data/kg_ctgan.csv'
            self.y_attr = 'Attack'
        elif self.name == 'octgan':
            dataset_path = 'data/octgan.csv'
            self.y_attr = 'Attack'
        elif self.name == 'pategan':
            dataset_path = 'data/pategan.csv'
            self.y_attr = 'Attack'
        elif self.name == 'tablegan':
            dataset_path = 'data/tablegan.csv'
            self.y_attr = 'Attack'
        elif self.name == 'ctgan_dp_gan':
            dataset_path = 'data/ctgan_dp_gan.csv'
            self.y_attr = 'Attack'
        else:
            raise ValueError(f'Dataset {self.name} is not part of the supported datasets')

        if params.attack_category == 'distributional_privacy_leakage':
            if self.name == 'Adult':
                dataset_path = 'data/Adult_10000.csv'
            elif self.name == 'GSS':
                dataset_path = 'data/GSS_5079.csv'
            else:
                raise ValueError(f'Dataset {self.name} is not part of the supported datasets for privacy leakage attack')


        df = pd.read_csv(dataset_path)
        # df = df[df[params.sensitive_attributes].notna()]
        self.sensitive_vals = {}
        self.sensitive_attributes = params.sensitive_attributes
        self.missing_nonsensitive_attributes = params.missing_nonsensitive_attributes
        self.missing_nsa_vals = {}
        # the y_labels name is confusing. need to change it later
        self.y_labels = y_labels_dict[self.name]
        self.ground_truths = {}
        for attr in params.sensitive_attributes:
            # self.sensitive_vals[attr] = df[attr].unique()
            self.sensitive_vals[attr] = self.y_labels[attr]
            self.ground_truths[attr] = df[attr].to_numpy()
        for attr in self.missing_nonsensitive_attributes:
            self.missing_nsa_vals[attr] = df[attr].unique()
        df = df.drop(columns=params.sensitive_attributes)
        df = df.drop(columns=self.missing_nonsensitive_attributes)
        # df = df.reset_index(inplace = True, drop = True)
        self.data = df.copy()