import os

from core.dvhcalculation import load

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# root_folder = r'C:\final_plans\ECLIPSE\ECPLIPSE_VMAT'
# root_folder = r'E:\COMPETITION 2017\final_plans\ECLIPSE\ECPLIPSE_VMAT'

root_folder = r'D:\Final_Plans\ECPLIPSE_VMAT'

# linux

# root_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans'
# dest_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans'

# dest_folder = r'C:\final_plans\ECLIPSE'

dest_folder = r'D:\Final_Plans'

#
# dest_folder = r'E:\COMPETITION 2017\final_plans\ECLIPSE'

# database_file = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/all_final_plans.hdf5'

# database_file = r"C:\final_plans\ECLIPSE\eclipse_vmat_dvh.hdf5"
# database_file = r"E:\COMPETITION 2017\final_plans\ECLIPSE\eclipse_vmat_dvh.hdf5"

database_file = r"D:\Final_Plans\all_final_plans.hdf5"

data = 'paper_constraints.xlsx'
sheet = 'competition_criteria_OAR'
data_path = os.path.join(DATA_DIR, data)

# low score plan
low_score = load(os.path.join(DATA_DIR, 'low_41_score_eclipse_vmat.dvh'))['DVH']
high_score = load(os.path.join(DATA_DIR, 'high_100_score_eclipse_vmat.dvh'))['DVH']
# pass
