import pandas as pd
import re
import numpy as np

"""
File to hold dosimetric 2. Dosimetric Criteria Sheet.pdf  from 2016 Competition Criteria;


"""
# Plan competition constrains
constrains_ptv_tot = {'D99': 99, 'D95': 95, 'D50': 50, 'Dcc': 0.3, 'HI': 5000, 'CI': 4750}
cont_heart = {'mean_value': 'mean', 'V15': 1500, 'D5': 5}
breast_right = {'Dcc': 0.3, 'D5': 5}
spinal_cord = {'Dcc': 0.3}
lr = {'V5': 500}
ll = {'mean_value': 'mean', 'V20': 2000, 'V10': 1000, 'V5': 500}
CTV_LUMPECTOMY = {'Dmax_position': 'max'}
BODY = {'Dmax_position': 'max'}

constrains = {'PTV_TOT_EVAL': constrains_ptv_tot,
              'HEART': cont_heart,
              'BREAST_RIGHT': breast_right,
              'SPINAL CORD': spinal_cord,
              'LUNG_RIGHT': lr,
              'LUNG_LEFT': ll,
              'BODY': BODY}
# 'CTV-LUMPECTOMY': CTV_LUMPECTOMY}

# Scores
scores_ptv_tot = {'D99': ['lower', [4500, 4750], [0, 15]],
                  'D95': ['lower', [4500, 5000], [0, 5]],
                  'D50': ['upper', [5200, 5400], [0, 5]],
                  'Dcc': ['upper', [5500, 5700], [0, 5]],
                  'HI': ['upper', [0.08, 0.12, 0.15, 0.2], [0, 1, 3, 5]],
                  'CI': ['lower', [0.6, 0.7, 0.8, 0.9], [0, 1, 4, 5]]}

scores_cont_heart = {'mean_value': ['upper', [400, 500], [0, 10]],
                     'V15': ['upper', [15, 20], [0, 5]],
                     'D5': ['upper', [2000, 2500], [0, 5]]}

scores_breast_right = {'Dcc': ['upper', [200, 300], [0, 2]],
                       'D5': ['upper', [200, 300], [0, 4]]}

scores_spinal_cord = {'Dcc': ['upper', [800, 2000], [0, 5]]}

scores_lr = {'V5': ['upper', [3, 6], [0, 5]]}

scores_ll = {'mean_value': ['upper', [900, 1500], [0, 5]],
             'V20': ['upper', [15, 20], [0, 5]],
             'V10': ['upper', [30, 40], [0, 5]],
             'V5': ['upper', [50, 60, 70], [0, 1, 4]]}

scores_BODY = {'Dmax_position': ['Dmax_position', [0, 1], [0, 5]]}

scores0 = {'PTV_TOT_EVAL': scores_ptv_tot,
           'HEART': scores_cont_heart,
           'BREAST_RIGHT': scores_breast_right,
           'SPINAL CORD': scores_spinal_cord,
           'LUNG_RIGHT': scores_lr,
           'LUNG_LEFT': scores_ll,
           'BODY': scores_BODY}


class Competition2016(object):
    def __init__(self):
        # Plan competition constrains
        self.constrains_ptv_tot = {'D99': 99, 'D95': 95, 'D50': 50, 'Dcc': 0.3, 'HI': 5000, 'CI': 4750}
        self.cont_heart = {'mean_value': 'mean', 'V15': 1500, 'D5': 5}
        self.breast_right = {'Dcc': 0.3, 'D5': 5}
        self.spinal_cord = {'Dcc': 0.3}
        self.lr = {'V5': 500}
        self.ll = {'mean_value': 'mean', 'V20': 2000, 'V10': 1000, 'V5': 500}
        self.CTV_LUMPECTOMY = {'Dmax_position': 'max'}
        self.BODY = {'Dmax_position': 'max'}

        self.constrains = {'PTV_TOT_EVAL': constrains_ptv_tot,
                           'HEART': cont_heart,
                           'BREAST_RIGHT': breast_right,
                           'SPINAL CORD': spinal_cord,
                           'LUNG_RIGHT': lr,
                           'LUNG_LEFT': ll,
                           'BODY': BODY}
        # 'CTV-LUMPECTOMY': CTV_LUMPECTOMY}

        # Scores
        self.scores_ptv_tot = {'D99': ['lower', [4500, 4750], [0, 15]],
                               'D95': ['lower', [4500, 5000], [0, 5]],
                               'D50': ['upper', [5200, 5400], [0, 5]],
                               'Dcc': ['upper', [5500, 5700], [0, 5]],
                               'HI': ['upper', [0.08, 0.12, 0.15, 0.2], [0, 1, 3, 5]],
                               'CI': ['lower', [0.6, 0.7, 0.8, 0.9], [0, 1, 4, 5]]}

        self.scores_cont_heart = {'mean_value': ['upper', [400, 500], [0, 10]],
                                  'V15': ['upper', [15, 20], [0, 5]],
                                  'D5': ['upper', [2000, 2500], [0, 5]]}

        self.scores_breast_right = {'Dcc': ['upper', [200, 300], [0, 2]],
                                    'D5': ['upper', [200, 300], [0, 4]]}

        self.scores_spinal_cord = {'Dcc': ['upper', [800, 2000], [0, 5]]}

        self.scores_lr = {'V5': ['upper', [3, 6], [0, 5]]}

        self.scores_ll = {'mean_value': ['upper', [900, 1500], [0, 5]],
                          'V20': ['upper', [15, 20], [0, 5]],
                          'V10': ['upper', [30, 40], [0, 5]],
                          'V5': ['upper', [50, 60, 70], [0, 1, 4]]}

        self.scores_BODY = {'Dmax_position': ['Dmax_position', [0, 1], [0, 5]]}

        self.scores = {'PTV_TOT_EVAL': scores_ptv_tot,
                       'HEART': scores_cont_heart,
                       'BREAST_RIGHT': scores_breast_right,
                       'SPINAL CORD': scores_spinal_cord,
                       'LUNG_RIGHT': scores_lr,
                       'LUNG_LEFT': scores_ll,
                       'BODY': scores_BODY}


def read_scoring_criteria(f):
    cGy = 100.0
    df = pd.read_csv(f, sep='\t')
    df = df[['Plan Quality Metric Component', 'Objective(s)', 'Max Score']].dropna()
    df['Max Score'] = df['Max Score'].apply(lambda x: x.split('*')[0]).astype(float)

    constrains_types = []
    for ob in df[['Objective(s)']].iterrows():
        if ob[1][0][0] == '<':
            constrains_types.append('upper')
        elif ob[1][0][0] == '>':
            constrains_types.append('lower')

    test = []
    for pq in df[['Plan Quality Metric Component']].iterrows():
        tmp = list(filter(None, re.split("\[(.*?)\].?", pq[1][0])))
        test.append(tmp)

    structures_names = []
    constrains_keys = []
    constrains_values = []
    for crit in test:
        if len(crit) == 1:
            structures_names.append('Global Max')
            constrains_keys.append('max')
            constrains_values.append('max')
        if len(crit) == 2:
            structures_names.append(crit[0])
            if 'Max' in crit[1]:
                constrains_keys.append('max')
                constrains_values.append('max')
            elif 'Mean' in crit[1]:
                constrains_keys.append('mean_value')
                constrains_values.append('mean')
        if len(crit) > 2:
            structures_names.append(crit[0])
            if len(crit) == 3:
                if 'Homogeneity' in crit[1]:
                    constrains_keys.append('HI')
                    cv = float(re.findall("\d+\.\d+", crit[2])[0])
                    cv *= cGy
                    constrains_values.append(cv)
                elif 'Conformation' in crit[1]:
                    constrains_keys.append('CI')
                    cv = float(re.findall("\d+\.\d+", crit[2])[0])
                    cv *= cGy
                    constrains_values.append(cv)
            elif 'cc' in crit[2]:
                constrains_keys.append('Dcc')
                constrains_values.append(float(re.findall("\d+\.\d+", crit[2])[0]))
            elif '%' in crit[2] and 'D' in crit[1]:
                cv = int(float(re.findall("\d+\.\d+", crit[2])[0]))
                constrains_keys.append('D' + str(cv))
                constrains_values.append(cv)
            elif '%' in crit[2] and 'V' in crit[1]:
                cv = int(float(re.findall("\d+\.\d+", crit[2])[0]))
                constrains_keys.append('V' + str(cv))
                constrains_values.append(cv)

    df['structures_name'] = structures_names
    df['constrain'] = constrains_keys
    df['constrain_value'] = constrains_values
    df['constrains_type'] = constrains_types

    # get objectives
    objectives = []
    obl = []
    obh = []
    for row in df[['Objective(s)']].iterrows():
        cval = np.asarray(re.findall("\d+\.?\d+", row[1].to_string()), dtype=float)
        cval.sort()
        if cval[0] > 1:
            # All doses to cGy
            cval *= cGy
        obl.append(cval[0])
        obh.append(cval[1])
        objectives.append(cval)

    scores_arr = []
    for row in df[['Max Score']].iterrows():
        score_i = np.array([0.0, row[1]])
        scores_arr.append(score_i)

    df['value_score'] = objectives
    df['value_low'] = obl
    df['value_high'] = obh
    df['scores_array'] = scores_arr

    dfi = df.set_index('structures_name')

    s_names = dfi.index.unique()
    constrains_all = {}
    scores_all = {}
    for name in s_names:
        df_tmp = dfi.loc[[name]]
        constrains_tmp = {}
        scores_tmp = {}
        for i in range(len(df_tmp.index)):
            key = df_tmp['constrain'].values[i]
            val = df_tmp['constrain_value'].values[i]
            constrains_tmp[key] = val

            c_type = df_tmp['constrains_type'].values[i]
            v_s = df_tmp['value_score'].values[i]
            s_a = df_tmp['scores_array'].values[i]
            scores_tmp[key] = [c_type, v_s, s_a]

        scores_all[name] = scores_tmp
        constrains_all[name] = constrains_tmp

    criteria = ['constrain', 'constrain_value', 'constrains_type', 'value_low', 'value_high', 'Max Score']

    return constrains_all, scores_all, dfi[criteria]


class Competition2017(object):
    def __init__(self, path_to_criteria):
        self.f = path_to_criteria

    def competition_criteria(self):
        read_scoring_criteria(self.f)


if __name__ == '__main__':
    f = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/PlanIQ Criteria TPS PlanIQ matched str names - TXT Fromat - Last mod Jan23.txt'
    cGy = 100.0
    df = pd.read_csv(f, sep='\t')
    df = df[['Plan Quality Metric Component', 'Objective(s)', 'Max Score']].dropna()
    df['Max Score'] = df['Max Score'].apply(lambda x: x.split('*')[0]).astype(float)

    constrains_types = []
    for ob in df[['Objective(s)']].iterrows():
        if ob[1][0][0] == '<':
            constrains_types.append('upper')
        elif ob[1][0][0] == '>':
            constrains_types.append('lower')

    test = []
    for pq in df[['Plan Quality Metric Component']].iterrows():
        tmp = list(filter(None, re.split("\[(.*?)\].?", pq[1][0])))
        test.append(tmp)

    structures_names = []
    constrains_keys = []
    constrains_values = []
    for crit in test:
        if len(crit) == 1:
            structures_names.append('Global Max')
            constrains_keys.append('max')
            constrains_values.append('max')
        if len(crit) == 2:
            structures_names.append(crit[0])
            if 'Max' in crit[1]:
                constrains_keys.append('max')
                constrains_values.append('max')
            elif 'Mean' in crit[1]:
                constrains_keys.append('mean_value')
                constrains_values.append('mean')
        if len(crit) > 2:
            structures_names.append(crit[0])
            if len(crit) == 3:
                if 'Homogeneity' in crit[1]:
                    constrains_keys.append('HI')
                    cv = float(re.findall("\d+\.\d+", crit[2])[0])
                    cv *= cGy
                    constrains_values.append(cv)
                elif 'Conformation' in crit[1]:
                    constrains_keys.append('CI')
                    cv = float(re.findall("\d+\.\d+", crit[2])[0])
                    cv *= cGy
                    constrains_values.append(cv)
            elif 'cc' in crit[2]:
                constrains_keys.append('Dcc')
                constrains_values.append(float(re.findall("\d+\.\d+", crit[2])[0]))
            elif '%' in crit[2] and 'D' in crit[1]:
                cv = int(float(re.findall("\d+\.\d+", crit[2])[0]))
                constrains_keys.append('D' + str(cv))
                constrains_values.append(cv)
            elif '%' in crit[2] and 'V' in crit[1]:
                cv = int(float(re.findall("\d+\.\d+", crit[2])[0]))
                constrains_keys.append('V' + str(cv))
                constrains_values.append(cv)

    df['structures_name'] = structures_names
    df['constrain'] = constrains_keys
    df['constrain_value'] = constrains_values
    df['constrains_type'] = constrains_types

    # get objectives
    objectives = []
    obl = []
    obh = []
    for row in df[['Objective(s)']].iterrows():
        cval = np.asarray(re.findall("\d+\.?\d+", row[1].to_string()), dtype=float)
        cval.sort()
        if cval[0] > 1:
            # All doses to cGy
            cval *= cGy
        obl.append(cval[0])
        obh.append(cval[1])
        objectives.append(cval)

    scores_arr = []
    for row in df[['Max Score']].iterrows():
        score_i = np.array([0.0, row[1]])
        scores_arr.append(score_i)

    df['value_score'] = objectives
    df['value_low'] = obl
    df['value_high'] = obh
    df['scores_array'] = scores_arr

    dfi = df.set_index('structures_name')

    s_names = dfi.index.unique()
    constrains_all = {}
    scores_all = {}
    for name in s_names:
        df_tmp = dfi.loc[[name]]
        constrains_tmp = {}
        scores_tmp = {}
        for i in range(len(df_tmp.index)):
            key = df_tmp['constrain'].values[i]
            val = df_tmp['constrain_value'].values[i]
            constrains_tmp[key] = val

            c_type = df_tmp['constrains_type'].values[i]
            v_s = df_tmp['value_score'].values[i]
            s_a = df_tmp['scores_array'].values[i]
            scores_tmp[key] = [c_type, v_s, s_a]

        scores_all[name] = scores_tmp
        constrains_all[name] = constrains_tmp

    criteria = ['constrain', 'constrain_value', 'constrains_type', 'value_low', 'value_high', 'Max Score']

    # dfi[criteria].to_excel('criteria_2017.xlsx')
