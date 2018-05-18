import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

dvh_file = os.path.join(DATA_DIR, 'example_differential_DVH_CC.txt')
models_file = os.path.join(DATA_DIR, 'models.inp')

dvh_diff_file = os.path.join(DATA_DIR, "RD_DIFF_DVH.dcm")
