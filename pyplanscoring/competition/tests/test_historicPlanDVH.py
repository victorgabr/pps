from unittest import TestCase

import os
from pyplanscoring.competition.statistical_dvh import HistoricPlanDVH


class TestHistoricPlanDVH(TestCase):
    def test_to_hdf(self):
        hist_data = HistoricPlanDVH(root_folder)
        hist_data.set_participant_folder()
        hist_data.load_dvh()

        dest_hdf = os.path.join(dest_folder, 'eclipse_vmat_dvh.hdf5')

        hist_data.to_hdf(dest_hdf)

    def test_load_hdf(self):
        dest_hdf = os.path.join(dest_folder, 'eclipse_vmat_dvh.hdf5')

        df = HistoricPlanDVH().load_hdf(dest_hdf)
