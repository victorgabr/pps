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

scores = {'PTV_TOT_EVAL': scores_ptv_tot,
          'HEART': scores_cont_heart,
          'BREAST_RIGHT': scores_breast_right,
          'SPINAL CORD': scores_spinal_cord,
          'LUNG_RIGHT': scores_lr,
          'LUNG_LEFT': scores_ll,
          'BODY': scores_BODY}
