import os
from unittest import TestCase

import pandas as pd

from competition.tests import root_folder, dest_folder
from pyplanscoring.core.types import DVHData
from pyplanscoring.competition.statistical_dvh import HistoricPlanDVH

database = 'photons_versus_protons.hdf5'


class TestHistoricPlanDVH(TestCase):
    def test_to_hdf(self):
        hist_data = HistoricPlanDVH(root_folder)
        hist_data.set_participant_folder()
        hist_data.load_dvh()

        dest_hdf = os.path.join(dest_folder, database)

        hist_data.to_hdf(dest_hdf)

        # # TODO encapsulate this at HistoricPlanDVH class
        # # generate database mappings
        # import re
        # import numpy as np
        # tmp = [f.split('/') for f in hist_data.dvh_files[0]]
        #
        # folders = [f[-3] for f in tmp]
        # part = [f[-2] for f in tmp]
        # score = []
        # for s in part:
        #     res = re.findall("\d+\.\d+", s)
        #     if res:
        #         score.append(res)
        #     else:
        #         score.append(s)
        #
        # assert len(folders) == len(part) and len(part) == len(score)
        #
        # import pandas as pd
        # df = pd.DataFrame(folders, columns=["Technique"])
        # df['Participant'] = part
        # df['score'] = np.array(score, dtype=float)
        # df['Path'] = hist_data.dvh_files[0]
        # # save to hdf as db mappings
        # df.to_hdf(dest_hdf, 'db')

    def test_load_hdf(self):
        dest_hdf = os.path.join(dest_folder, 'photons_versus_protons.hdf5')

        df = HistoricPlanDVH().load_hdf(dest_hdf)

        db_df = pd.read_hdf(dest_hdf, 'db')

        vf_data = {}
        for s in df.columns:
            doses = []
            volume = []
            for row in df.iterrows():
                dvh = DVHData(row[1][s])
                doses.append(dvh.dose_focused_format)
                volume = dvh.volume_focused_format

            vf_data[s] = pd.DataFrame(doses, columns=volume)

        # fix all plans
        df_temp = df.copy()

        # b = df_temp['BRACHIAL PLEXU']
        # a = df_temp['BRACHIAL PLEXUS']
        # mask_a = a.isnull()
        # mask_b = b.isnull()
        #
        # a[mask_a] = b[~mask_b].copy()

        # pairs

        # pairs = [('BRACHIAL PLEXUS', 'BRACHIAL PLEXU'),
        #          ('OPTIC CHIASM PRV', 'OPTIC CHIASM P'),
        #          ('OPTIC N. LT PRV', 'OPTIC N. LT PR'),
        #          ('OPTIC N. RT PRV', 'OPTIC N. RT PR'),
        #          ('PTV63-BR.PLX 1MM', 'PTV63-BR.PLX 1'),
        #          ('PTV70-BR.PLX 4MM', 'PTV70-BR.PLX 4'),
        #          ('SPINAL CORD PRV', 'SPINAL CORD PR')]
        #
        # for pair in pairs:
        #     b = df_temp[pair[0]]
        #     a = df_temp[pair[1]]
        #     mask_a = a.isnull()
        #     mask_b = b.isnull()
        #
        #     a[mask_a] = b[~mask_b].copy()
        #     df_temp = df_temp.drop(pair[1], axis=1)
        #
        # key = HistoricPlanDVH().__class__.__name__
        # df_temp.to_hdf(dest_hdf, key)
