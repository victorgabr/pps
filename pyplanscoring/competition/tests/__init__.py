import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

root_folder = r'E:\COMPETITION 2017\final_plans\ECLIPSE\ECPLIPSE_VMAT'
dest_folder = r'E:\COMPETITION 2017\final_plans\ECLIPSE'
database_file = r"E:\COMPETITION 2017\final_plans\ECLIPSE\eclipse_vmat_dvh.hdf5"
data = 'paper_constraints.xlsx'
sheet = 'competition_constraints'
data_path = os.path.join(DATA_DIR, data)
