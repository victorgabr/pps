"""

H&N-Nasopharynx - Case

Protocol: The dosimetric criteria are based on the Ontario H&N IMRT Protocol
Targets & Doses: PTV70 (70 Gy), PTV63 (63 Gy), and PTV56 (56Gy)
Fractionation Schemes: Simultaneous Integrated Boost (SIB), with 35 fractions


references: https://www.sciencedirect.com/science/article/pii/S0167814013002193

The model-based approach

The model-based approach consists of two consecutive phases:
phase a , aiming at the selection of patients who may benefit from
protons, and phase b, aiming at the clinical validation of proton
therapy by so-called sequential prospective observational cohort
(SPOC) studies using appropriate historical comparisons as a refer-
ence or by RCT’s in selected situations.

Phase a : model-based indications

Phase a of the model-based approach consists of 3 steps,
including: (1) the development and validation of Normal Tissue
Complication Probability (NTCP) models in patients treated with
state-of-the-art photon radiotherapy; (2) individual in silico plan-
ning comparative studies [21], and (3): estimation of the potential
benefit of the new radiation technique in reducing side effects by
integrating the results of ISPC into NTCP-models. The main purpose
of these 3 steps is to select patients that will most likely benefit from
protons compared to photons in terms of NTCP-value reductions.


Step 1: NTCP models
Step 2: in silico planning comparative (ISPC) studies
Step 3: estimation of the clinical benefit

"""
import os

from sklearn.manifold import TSNE

from radiobiology.ntcp_models import NTCPLKBModel


class ISPC:
    pass


import numpy as np
import scipy.stats as st


def t_welch(x, y, tails=2):
    """Welch's t-test for two unequal-size samples, not assuming equal variances
        https://gist.github.com/jdmonaco/5922991
    """
    assert tails in (1, 2), "invalid: tails must be 1 or 2, found %s" % str(tails)

    x, y = np.asarray(x), np.asarray(y)
    nx, ny = x.size, y.size
    vx, vy = x.var(), y.var()
    df = int((vx / nx + vy / ny) ** 2 /  # Welch-Satterthwaite equation
             ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)))
    t_obs = (x.mean() - y.mean()) / np.sqrt(vx / nx + vy / ny)
    p_value = tails * st.t.sf(abs(t_obs), df)
    return t_obs, p_value


if __name__ == '__main__':
    # selection of plans
    from competition.statistical_dvh import StatisticalDVH, PlanningItemDVH, GeneralizedEvaluationMetricWES, \
        HistoricPlanDVH
    from competition.tests import database_file, data_path, sheet, dest_folder
    import pandas as pd
    import matplotlib.pyplot as plt

    stats_dvh = StatisticalDVH()
    stats_dvh.load_data_from_hdf(database_file)

    dvh_df = stats_dvh.dvh_data.copy()
    db_df = stats_dvh.db_df.copy()

    ctr = pd.read_excel(data_path, sheet)
    gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, ctr)
    gem_wes_obj.load_constraints_stats(database_file, sheet)
    ctr_stats = gem_wes_obj.constraints_stats

    #  calc NTCP Int. J. Radiation Oncology Biol. Phys., Vol. 78, No. 2, pp. 449–453, 2010
    TD50 = 39.9
    m = 0.40
    n = 1.0

    structure_name = 'PAROTID LT'
    wes = []
    gem_wes = []
    plan_gem_wes = []
    ntcps = []
    for row in range(len(db_df)):
        print('start plan: ', row)
        dvh_i = dvh_df.iloc[row]
        parotid_dvh = dvh_i[structure_name]
        # NTCP calc
        # calculating
        ntcp_calc = NTCPLKBModel(parotid_dvh, [TD50, m, n])
        ntcp = ntcp_calc.calc_model()
        ntcps.append(ntcp)
        pi_t = PlanningItemDVH(plan_dvh=dvh_i)
        gem_wesi = gem_wes_obj.get_gem_wes(pi_t, structure_name, 'Mean[Gy] <= 26')
        wesi = gem_wes_obj.weighted_cumulative_probability(row, structure_name)
        p_gem_wes = gem_wes_obj.calc_plan_gem_wes(pi_t)
        plan_gem_wes.append(p_gem_wes)
        wes.append(wesi)
        gem_wes.append(gem_wesi)
        print('end plan: ', row)

    db_df['WES'] = wes
    db_df['GEM_WES'] = gem_wes
    db_df['PLAN_GEM_WES'] = plan_gem_wes
    db_df['NTCP'] = ntcps

    db_df['Technique'] = db_df['Technique'].apply(lambda x: "VMAT" if x != "IMPT" else "IMPT")

    ranking_ntcp = db_df.sort_values('NTCP').drop("Path", axis=1)  #
    ranking_wes = db_df.sort_values('WES').drop("Path", axis=1)
    ranking_gem_wes = db_df.sort_values('GEM_WES').drop("Path", axis=1)
    ranking_plan_em_wes = db_df.sort_values('PLAN_GEM_WES').drop("Path", axis=1)
    dest = os.path.join(dest_folder, 'ranking_gem_wes_mean_dose.hdf5')
    # db_df.to_hdf(dest, "ranking")

    ranking = pd.read_hdf(os.path.join(dest_folder, 'ranking_gem_wes_mean_dose.hdf5'), "ranking")
    ranking_ntcp = ranking.sort_values('NTCP').drop("Path", axis=1)

    plt.style.use('ggplot')

    plt.plot(ranking_ntcp['NTCP'], ranking_ntcp['GEM_WES'], '.')
    plt.title("PAROTID LT - GEM_WES versus NTCP")
    plt.xlabel('NTCP')
    plt.ylabel('GEM_WES')
    plt.text(0.8, .5, "Kendall's tau: 0.848")
    ranking_ntcp[['NTCP', 'GEM_WES']].corr(method='kendall')

    # mask_impt = ranking_ntcp['Technique'] == "IMPT"
    #
    # impt = ranking_gem_wes['PLAN_GEM_WES'].loc[ranking_gem_wes['Technique'] == 'IMPT']
    # vmat = ranking_gem_wes['PLAN_GEM_WES'].loc[ranking_gem_wes['Technique'] != 'IMPT']
    #
    # impt_ntcp = ranking_ntcp['NTCP'].loc[ranking_ntcp['Technique'] == 'IMPT']
    # vmat_ntcp = ranking_ntcp['NTCP'].loc[ranking_ntcp['Technique'] != 'IMPT']
    #
    # # statistical tests calculate pVALUE
    #
    # result = t_welch(impt, vmat)
    # print(result)
    # result1 = st.ttest_ind(impt, vmat, equal_var=False)
    # ntcp_result = st.ttest_ind(impt_ntcp, vmat_ntcp)
    #

    #
    # # todo Plot T-sne proton x vmat prototypes
    #
    # ranking = pd.read_hdf(os.path.join(dest_folder, 'ranking_gem_wes.hdf5'), "ranking")
    # ranking = ranking.sort_values('PLAN_GEM_WES').drop("Path", axis=1)
    #
    # # todo plot plan DVH at experience
    #
    # jet = plt.get_cmap('jet')
    # colors = iter(jet(np.linspace(0, 1, 10)))
    #
    # for i in range(10):
    #     idx = ranking.index[i]
    #     tech = ranking.loc[idx]["Technique"]
    #     fig, ax = stats_dvh.plot_historical_dvh(structure_name, "Dose", "Volume", structure_name)
    #     p = stats_dvh.get_plan_dvh(idx)[structure_name]
    #     pdvh = p['data'] / p['data'][0] * 100
    #     ax.plot(pdvh, label="Rank %s - %s" % (str(i), tech), color=next(colors))
    #     ax.legend()
    #
    # ctr = pd.read_excel(data_path, sheet)
    # gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, ctr)
    # gem_wes_obj.load_constraints_stats(database_file, sheet)
    # ctr_stats = gem_wes_obj.constraints_stats
    #
    # projected = TSNE().fit_transform(ctr_stats.T)
    # n_ranking = -1
    # mask = ranking['Technique'] == "IMPT"
    # mask = mask[:n_ranking]
    # data = projected[:n_ranking]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(data[:, 0][mask], data[:, 1][mask], 'g.', label='IMPT')
    # ax.plot(data[:, 0][~mask], data[:, 1][~mask], 'r.', label='VMAT')
    # i = 1
    # for xy in zip(data[:, 0], data[:, 1]):  # <--
    #     ax.annotate('%s' % i, xy=xy, textcoords='data')  # <--
    #     i += 1
    #
    # ax.legend()
    # plt.show()
    #
    # # structure DVH
    # structure_name = 'PAROTID LT'
    # parotid_dvh_vf = stats_dvh.vf_data[structure_name]
    #
    # projected1 = TSNE().fit_transform(parotid_dvh_vf)
    # # protons
    # plt.figure()
    # n_ranking = -1
    # mask = ranking['Technique'] == "IMPT"
    # mask = mask[:n_ranking]
    # data1 = projected1[:n_ranking]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # ax.plot(data1[:, 0][mask], data1[:, 1][mask], 'g*', label='IMPT')
    # ax.plot(data1[:, 0][~mask], data1[:, 1][~mask], 'r.', label='VMAT')
    # ax.legend()
    # plt.show()
    # # i = 1
    # # for xy in zip(data1[:, 0], data1[:, 1]):  # <--
    # #     ax.annotate('%s' % i, xy=xy, textcoords='data')  # <--
    # #     i += 1
    # # plt.show()
