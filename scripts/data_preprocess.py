import pandas as pd

class DataPreprocessor:
    def __init__(self, input_file, output_file, reduced=False):
        self.input_file = input_file
        self.output_file = output_file
        self.reduced = reduced

    def preprocess_data(self):
        data = pd.read_excel(self.input_file)

        if self.reduced:
            data = self._preprocess_reduced(data)
        else:
            data = self._preprocess_full(data)

        data.to_csv(self.output_file, index=False)

    def _preprocess_reduced(self, data):
        data = data.drop(columns=['Index', 'Name', 'Ref', 'Ref_index'])

        cm_type_replacements = {
            'CNF': '1', 'CC': '1', 'CNT': '1', 'MWCNT': '1',
            'Graphene': '2', 'GO': '2', 'RGO': '2',
            'derived carbon': '3', 'derived carbon & CNT': '3'
        }
        data['CM_type'] = data['CM_type'].replace(cm_type_replacements)

        cm_morph_replacements = {
            '1D fibers': '1', '1D tubes': '1',
            '2D nanosheets': '2',
            '3D porous': '3', '3D networks': '3',
            'nanoparticles': '4', 'nanospheres': '4', 'rods': '4', 'tubes': '4', 'flower-like': '4'
        }
        data['CM_morph'] = data['CM_morph'].replace(cm_morph_replacements)

        ms2_morph_replacements = {
            'bulk': '0',
            'nanosheets': '1',
            'flower-like clusters': '2',
            'irregular nanoparticles': '3',
            'nanoparticles': '4', 'octahedron': '4', 'cubes': '4', 'rods': '4', 'spheres': '4',
            'hollow spheres': '5', 'hollow rods': '5', 'hollow cubes': '5', 'hollow nanoparticles': '5', 'double shell spheres': '5', 'yolk-shell spheres': '5'
        }
        data['MS2_morph'] = data['MS2_morph'].replace(ms2_morph_replacements)

        cp_morph_replacements = {
            'supported': '1',
            'embedded': '2',
            'coated': '3',
            'interconnected': '4'
        }
        data['CP_morph'] = data['CP_morph'].replace(cp_morph_replacements)

        return data

    def _preprocess_full(self, data):
        data = data.drop(columns=['Index', 'Name', 'CP', 'P_low', 'P_high', 'Ref', 'Ref_index'])

        boolean_columns = ['Ti', 'V', 'Fe', 'Co', 'Ni', 'Zr', 'Mo', 'Sn', 'W']
        data[boolean_columns] = data[boolean_columns].astype(bool)

        cm_type_replacements = {
            'derived carbon': 'derived carbon-based',
            'derived carbon & CNT': 'derived carbon-based',
            'CNT': 'CNT',
            'MWCNT': 'CNT',
            'CNF': 'CNF',
            'CC': 'CNF',
            'Graphene': 'G-based',
            'GO': 'G-based',
            'RGO': 'G-based'
        }
        data['CM_type'] = data['CM_type'].replace(cm_type_replacements)

        cm_morph_replacements = {
            'OD QDs': '0D',
            '2D nanosheets': '2D',
            '3D porous': '3D porous',
            '3D networks': '3D porous',
            'nanoparticles': '3D special',
            'nanospheres': '3D special',
            'rods': '3D special',
            'tubes': '3D special',
            'flower-like': '3D special'
        }
        data['CM_morph'] = data['CM_morph'].replace(cm_morph_replacements)

        ms2_morph_replacements = {
            'nanosheets': 'nanosheets',
            'flower-like clusters': 'flower-like clusters',
            'irregular nanoparticles': 'irregular nanoparticles',
            'hollow spheres': 'hollow morph',
            'hollow rods': 'hollow morph',
            'hollow cubes': 'hollow morph',
            'hollow nanoparticles': 'hollow morph',
            'double shell spheres': 'hollow morph',
            'yolk-shell spheres': 'hollow morph',
            'core-shell nanoparticles': 'hollow morph',
            'nanoparticles': 'nanoparticles',
            'octahedron': 'nanoparticles',
            'cubes': 'nanoparticles',
            'rods': 'nanoparticles',
            'spheres': 'nanoparticles'
        }
        data['MS2_morph'] = data['MS2_morph'].replace(ms2_morph_replacements)

        categorical_columns = ['CM_type', 'CM_morph', 'CP_morph']
        data[categorical_columns] = data[categorical_columns].replace(0, 'NoCP')

        data['SSA'] = data['SSA'].round(1)
        data['Cs'] = data['Cs'].round(0)

        return data
