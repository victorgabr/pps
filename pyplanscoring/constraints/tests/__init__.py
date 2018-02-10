import os

from constraints.metrics import PlanningItem
from core.dicom_reader import PyDicomParser

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# plot flag
plot_flag = False

rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')
rp = os.path.join(DATA_DIR, 'RP.dcm')

rs_dcm = PyDicomParser(filename=rs)
rd_dcm = PyDicomParser(filename=rd)
rp_dcm = PyDicomParser(filename=rp)

rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')

# planning item
pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)

structures_dict = rs_dcm.GetStructures()
pass
