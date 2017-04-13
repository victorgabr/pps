import logging
import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd

from pyplanscoring.competition.robust import mean, std
# todo write stats analysis from constrains results
from pyplanscoring.core.dicomparser import ScoringDicomParser

logger = logging.getLogger('results.py')

logging.basicConfig(filename='results.log', level=logging.DEBUG)


def get_competitions_results(root_folder, column='Raw Score'):
    error = {}
    results = []
    scores = {}
    index = ''
    idx = 0
    max_score = []
    for folder in os.listdir(root_folder):
        participant_folder = osp.join(root_folder, folder)

        files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name in
                 files if name.strip().endswith('.xlsx')]

        if files:
            if len(files) > 1:
                error[participant_folder] = files

            elif len(files) == 1:
                tmp = pd.read_excel(files[0], header=31).dropna().reset_index()
                header = pd.read_excel(files[0], header=30).dropna().reset_index().columns[1].split(',')
                res = tmp[column]
                raw_score = tmp['Raw Score'].sum()
                res.name = folder
                scores[folder] = [header, raw_score]
                results.append(res)

                if idx == 0:
                    index = tmp['index'].map(str) + '_' + tmp['constrain'].map(str)
                    max_score = tmp['Max Score']

        idx += 1

    df = pd.concat(results, axis=1)
    df.index = index

    scores = pd.DataFrame.from_dict(scores).T
    scores['name'] = scores[0].apply(lambda x: x[0].strip())
    scores['TPS'] = scores[0].apply(lambda x: x[1].strip())
    scores['Technique'] = scores[0].apply(lambda x: x[2].strip())
    scores['Plan Type'] = scores[0].apply(lambda x: x[3].strip())
    scores['Final or Trial'] = scores[0].apply(lambda x: x[4].strip())

    return df, scores, max_score


def plot_results(df):
    for i in df.index:
        plt.figure()
        val = df.loc[i].values
        plt.xlabel('Score')
        # sanitize CI and HI values
        if '_CI' in i:
            mask = val < 1
            val = val[mask]
            plt.xlabel('Score')
        if '_HI' in i:
            mask = val < 1
            val = val[mask]
            plt.xlabel('Score')

        m = mean(val)
        st = std(val)
        # xlim = [m - 4 * st, m + 4 * st]

        plt.hist(val)
        # plt.xlim(xlim)
        plt.ylabel('Number of Plans')
        plt.title(i)
        # fname = 'Raw_Score_ ' + i + '.png'
        # dest = osp.join(FIGURES_DIR, fname)
        # plt.savefig(dest, format='png', dpi=100)
        # plt.close('all')
        # plt.show()


def save_hist(df, column, figures_dir):
    for i in df.index:
        plt.figure()
        val = df.loc[i].values
        if column == 'Result':
            plt.xlabel('Dose [cGy]')
        else:
            plt.xlabel(column)

        # sanitize CI and HI values
        if '_CI' in i:
            mask = val < 1
            val = val[mask]
            plt.xlabel(column)
        if '_HI' in i:
            mask = val < 1
            val = val[mask]
            plt.xlabel(column)

        plt.hist(val)
        plt.ylabel('Number of Plans')
        plt.title(i)
        fname = column + '_' + i + '.png'
        dest = osp.join(figures_dir, fname)
        plt.savefig(dest, format='png', dpi=100)
        plt.close('all')

    FIGURES_DIR = osp.join(
        osp.dirname(osp.realpath(__file__)),
        'figures',
    )


def sort_reports():
    plt.style.use('ggplot')
    column = "Result"
    root_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/plans/submited_plans/reports'
    df, scores, max_score = get_competitions_results(root_folder, column=column)

    mask = scores['Final or Trial'] == 'Final Plan'

    # Get final score

    final_score = scores.loc[mask]
    eclipse = final_score[final_score['TPS'] == 'Elekta-XiO']
    eclipse[eclipse['Plan Type'] == 'Clinical Plan'].reset_index().set_index(1).sort_index()


if __name__ == '__main__':

    root_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/plans/submited_plans/plans'

    # parse RT plan files

    participant = {}
    for folder in os.listdir(root_folder):
        participant_folder = osp.join(root_folder, folder)
        print('-----------')
        print('Folder: %s' % folder)
        files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name in
                 files if name.strip().endswith('.dcm')]

        plan_files = []
        for f in files:
            print('file: %s' % f)
            try:
                obj = ScoringDicomParser(filename=f)
                rt_type = obj.GetSOPClassUID()
                if rt_type == 'rtplan':
                    plan_files.append(f)

            except:
                logger.exception('Error in file %s' % f)

        participant[folder] = plan_files

        # process plan files
