import os
import platform

from pyplanscoring.dosimetric import read_scoring_criteria
from pyplanscoring.scoring import get_participant_folder_data, Participant

__version__ = '0.0.1'
__author__ = 'Dr. Victor Gabriel Leandro Alves, D.Sc.'
__copyright__ = "Copyright (C) 2004 Victor Gabriel Leandro Alves"
__license__ = "Licenced for evaluation purposes only"

txt = "PlanReport - H&N Nasopharynx - 2017 RT Plan Competition: %s \n" \
      "Be the strongest link in the radiotherapy chain\n" \
      "https://radiationknowledge.org \n" \
      "Author: %s\n" \
      "Copyright (C) 2017 Victor Gabriel Leandro Alves, All rights reserved\n" \
      "Platform details: Python %s on %s\n" \
      "This program aims to calculate an approximate score only.\n" \
      "your final score may be different due to structure boundaries and dose interpolation uncertainties\n" \
      "%s" \
      % (__version__, __author__, platform.python_version(), platform.system(), __license__)


def main():
    print(txt)
    wd = os.getcwd()
    dicom_dir = os.path.join(wd, 'dicom_files')
    participant_name = 'Plan_report'
    truth, files_data = get_participant_folder_data(participant_name, dicom_dir)
    if truth:
        print(str('Loaded Plan Files:'))
        f_data = [os.path.split(f)[1] for f in files_data.index]
        print(f_data[0])
        print(f_data[1])
        print(f_data[2])
    else:
        msg = "Missing Dicom Files: " + files_data.to_string() + '\n' \
                                                                 'Insert RT/RS/RP into the folder dicom_files '

        print(msg)

    # reading competition criteria
    f_2017 = os.path.join(wd, 'Scoring Criteria.txt')
    constrains, scores, criteria = read_scoring_criteria(f_2017)
    rd = files_data.reset_index().set_index(1).ix['rtdose']['index']
    rp = files_data.reset_index().set_index(1).ix['rtplan']['index']
    rs = files_data.reset_index().set_index(1).ix['rtss']['index']

    print('------------- Calculating DVH and score --------------')
    participant = Participant(rp, rs, rd, upsample='_up_sampled_', end_cap=True)
    participant.set_participant_data(participant_name)
    val = participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria, dicom_dvh=True)

    print('Plan Score: %1.3f' % val)
    out_file = os.path.join(dicom_dir, 'plan_scoring_report.xls')
    banner_path = os.path.join(wd, '2017 Plan Comp Banner.jpg')
    participant.save_score(out_file, banner_path=banner_path)
    print('Report saved: %s' % out_file)
    input("Press enter to exit.")


if __name__ == '__main__':
    main()
