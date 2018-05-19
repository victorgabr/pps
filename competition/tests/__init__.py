import os

from pyplanscoring.core.dvhcalculation import load

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# linux

root_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/photon_versus_proton'
dest_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/photon_versus_proton'
database_file = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/photon_versus_proton/VMAT_versus_IMPT.hdf5'

data = 'paper_constraints.xlsx'
sheet = 'Ontario_HN_OAR'
data_path = os.path.join(DATA_DIR, data)

# low score plan
low_score = load(os.path.join(DATA_DIR, 'low_41_score_eclipse_vmat.dvh'))['DVH']
high_score = load(os.path.join(DATA_DIR, 'high_100_score_eclipse_vmat.dvh'))['DVH']
# pass


#  todo selection of plans
# from competition.statistical_dvh import HistoricPlanDVH, StatisticalDVH, GeneralizedEvaluationMetric, \
#     PlanningItemDVH
# from competition.tests import database_file, data_path, dest_folder
# import pandas as pd
#
# stats_dvh = StatisticalDVH()
# stats_dvh.load_data_from_hdf(database_file)
#
# dvh_df = stats_dvh.dvh_data.copy()
# db_df = stats_dvh.db_df.copy()
#
# # select passed plans
# sheet = "must_pass"
# ctr = pd.read_excel(data_path, sheet)
#
# gem = GeneralizedEvaluationMetric(stats_dvh, ctr)
# ctr_stats = gem.calc_constraints_stats()
#
# # criteria
# c0 = ctr_stats.loc[0] >= 63.0
# c1 = ctr_stats.loc[1] >= 56.7
# c2 = ctr_stats.loc[2] >= 50.4
# t1 = pd.np.logical_and(c0, c1)
# t = pd.np.logical_and(t1, c2)
#
# # data that passed PTV minimum coverage
# dvh_df = stats_dvh.dvh_data.loc[t]
# db_df = stats_dvh.db_df.loc[t]
#
# dest = os.path.join(dest_folder, 'VMAT_versus_IMPT.hdf5')
# dvh_df.to_hdf(dest, 'HistoricPlanDVH')
#
# db_df.to_hdf(dest, 'db')
