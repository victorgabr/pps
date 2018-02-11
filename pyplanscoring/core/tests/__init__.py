import os

import quantities as pq

from core.dicom_reader import PyDicomParser
from core.types import Dose3D

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# plot flag
plot_flag = True

rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')
rp = os.path.join(DATA_DIR, 'RP.dcm')

rs_dcm = PyDicomParser(filename=rs)
rd_dcm = PyDicomParser(filename=rd)
rp_dcm = PyDicomParser(filename=rp)

structures = rs_dcm.GetStructures()

# dose 3D

dose_values = rd_dcm.get_dose_matrix()
grid = rd_dcm.get_grid_3d()
dose_3d = Dose3D(dose_values, grid, pq.Gy)

str_names = ['LENS LT',
             'PAROTID LT',
             'BRACHIAL PLEXUS',
             'OPTIC N. RT PRV',
             'OPTIC CHIASM PRV',
             'OPTIC N. RT',
             'ORAL CAVITY',
             'BRAINSTEM',
             'SPINAL CORD',
             'OPTIC CHIASM',
             'LENS RT',
             'LARYNX',
             'SPINAL CORD PRV',
             'EYE LT',
             'PTV56',
             'BRAINSTEM PRV',
             'PTV70',
             'OPTIC N. LT PRV',
             'EYE RT',
             'PTV63',
             'OPTIC N. LT',
             'LIPS',
             'ESOPHAGUS',
             'PTV70']

to_index = {v['name']: k for k, v in structures.items()}

ptv70 = structures[to_index['PTV70']]
lens = structures[to_index['LENS LT']]
spinal_cord = structures[to_index['SPINAL CORD']]
parotid_lt = structures[to_index['PAROTID LT']]
brain = structures[6]
body = structures[4]
