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


class ISPC:
    pass


if __name__ == '__main__':
    # selection of plans
    from competition.statistical_dvh import StatisticalDVH, PlanningItemDVH, GeneralizedEvaluationMetricWES
    from competition.tests import database_file, data_path, sheet, dest_folder
    import pandas as pd

    stats_dvh = StatisticalDVH()
    stats_dvh.load_data_from_hdf(database_file)

    dvh_df = stats_dvh.dvh_data.copy()
    db_df = stats_dvh.db_df.copy()

    # select passed plans

    ctr = pd.read_excel(data_path, sheet)
    # gem = PopulationBasedGEM(stats_dvh, ctr)
    #
    # gem_plans = []
    # gemp = []
    # for row in range(len(db_df)):
    #     dvh_i = dvh_df.iloc[row]
    #     pi_t = PlanningItemDVH(plan_dvh=dvh_i)
    #     gem_t = gem.calc_gem(pi_t)
    #     # gem_p = gem_pop.calc_gem(pi_t)
    #     gem_plans.append(gem_t)
    #     # gemp.append(gem_p)
    #
    # db_df['GEM'] = gem_plans
    # # db_df['GEM_pop'] = gemp
    #
    # plt.plot(db_df['score'], db_df['GEM'], '.')

    # ranking = db_df.sort_values('GEM').drop("Path", axis=1)
    # ranking_pop = db_df.sort_values('GEM_pop').drop("Path", axis=1)
    # removing outliers of fake planes

    # rank now using WES
    # reseting index
    db_df = db_df.reset_index()
    dvh_df = dvh_df.reset_index()

    gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, ctr)
    gem_wes_obj.load_constraints_stats(database_file, sheet)
    structure_name = 'PAROTID LT'
    wes = []
    gem_wes = []
    plan_gem_wes = []
    for row in range(len(db_df)):
        print('start plan: ', row)
        dvh_i = dvh_df.iloc[row]
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

    ranking_wes = db_df.sort_values('WES').drop("Path", axis=1)
    ranking_gem_wes = db_df.sort_values('GEM_WES').drop("Path", axis=1)
    ranking_plan_em_wes = db_df.sort_values('PLAN_GEM_WES').drop("Path", axis=1)
    dest = os.path.join(dest_folder, 'ranking_gem_wes.hdf5')
    ranking_plan_em_wes.to_hdf(dest, 'plan_gem_wes')

    tech_ranking = ranking_plan_em_wes[['Technique', 'PLAN_GEM_WES']]

    ranking_per_technique = tech_ranking.round(2)
