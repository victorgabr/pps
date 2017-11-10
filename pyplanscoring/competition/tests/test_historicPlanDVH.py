import os
from unittest import TestCase

from competition.tests import root_folder, dest_folder
from pyplanscoring.competition.statistical_dvh import HistoricPlanDVH


class TestHistoricPlanDVH(TestCase):
    def test_to_hdf(self):
        hist_data = HistoricPlanDVH(root_folder)
        hist_data.set_participant_folder()
        hist_data.load_dvh()

        dest_hdf = os.path.join(dest_folder, 'all_final_plans.hdf5')

        hist_data.to_hdf(dest_hdf)

    def test_load_hdf(self):
        dest_hdf = os.path.join(dest_folder, 'all_final_plans.hdf5')

        df = HistoricPlanDVH().load_hdf(dest_hdf)

        # fix all plans
        df_temp = df.copy()

        # b = df_temp['BRACHIAL PLEXU']
        # a = df_temp['BRACHIAL PLEXUS']
        # mask_a = a.isnull()
        # mask_b = b.isnull()
        #
        # a[mask_a] = b[~mask_b].copy()

        # pairs

        pairs = [('BRACHIAL PLEXUS', 'BRACHIAL PLEXU'),
                 ('OPTIC CHIASM PRV', 'OPTIC CHIASM P'),
                 ('OPTIC N. LT PRV', 'OPTIC N. LT PR'),
                 ('OPTIC N. RT PRV', 'OPTIC N. RT PR'),
                 ('PTV63-BR.PLX 1MM', 'PTV63-BR.PLX 1'),
                 ('PTV70-BR.PLX 4MM', 'PTV70-BR.PLX 4'),
                 ('SPINAL CORD PRV', 'SPINAL CORD PR')]

        for pair in pairs:
            b = df_temp[pair[0]]
            a = df_temp[pair[1]]
            mask_a = a.isnull()
            mask_b = b.isnull()

            a[mask_a] = b[~mask_b].copy()
            df_temp = df_temp.drop(pair[1], axis=1)

        key = HistoricPlanDVH().__class__.__name__
        df_temp.to_hdf(dest_hdf, key)
