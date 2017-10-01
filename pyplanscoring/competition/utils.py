import configparser
import datetime
import logging
import os
import os.path as osp
import shutil
import smtplib
import time
import urllib
import urllib.request
from collections import OrderedDict
from email import encoders as Encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from xlsxwriter.utility import xl_rowcol_to_cell

from pyplanscoring.competition.report_generator import CompetitionReportPDF, FinalReportPDF
from pyplanscoring.core.dicomparser import ScoringDicomParser
from pyplanscoring.core.dosimetric import read_scoring_criteria
from pyplanscoring.core.dvhcalculation import load
from pyplanscoring.core.scoring import Participant

logger = logging.getLogger('utils.py')

logging.basicConfig(filename='Generate_reports.log', level=logging.DEBUG)


def save_formatted_report(df, out_file, banner_path=None, report_header=''):
    """
        Save an formated report using pandas and xlsxwriter
    :param df: Results dataframe
    :param out_file: filename path
    :param banner_path: banner path
    """
    start_row = 31
    number_rows = len(df.index)
    writer = pd.ExcelWriter(out_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='report', startrow=start_row)

    # Get access to the workbook and sheet
    workbook = writer.book
    worksheet = writer.sheets['report']

    # Reduce the zoom a little
    worksheet.set_zoom(65)
    # constrain_fmt = workbook.add_format({'align': 'center'})
    constrain_fmt = workbook.add_format({'align': 'center'})

    # # Total formatting
    number_format = workbook.add_format({'align': 'right', 'num_format': '0.00'})
    # # Total percent format
    total_percent_fmt = workbook.add_format({'align': 'right', 'num_format': '0.0%', 'bold': True})

    # Add a format. Light red fill with dark red text.
    format1 = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})

    # Add a format. Green fill with dark green text.
    format2 = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})

    # Format the columns by width and include number formats

    # Structure name
    nr = number_rows + start_row
    sname = "A2:A{}".format(nr + 1)
    worksheet.set_column(sname, 24)
    # constrain
    constrain = "B2:B{}".format(nr + 1)
    worksheet.set_column(constrain, 20, constrain_fmt)

    # constrain value
    constrain_value = "C2:C{}".format(nr + 1)
    worksheet.set_column(constrain_value, 20, constrain_fmt)

    # constrain type
    constrain_type = "D2:D{}".format(nr + 1)
    worksheet.set_column(constrain_type, 20, constrain_fmt)

    worksheet.conditional_format(constrain_type, {'type': 'text',
                                                  'criteria': 'containing',
                                                  'value': 'upper',
                                                  'format': format1})

    # Highlight the bottom 5 values in Red
    worksheet.conditional_format(constrain_type, {'type': 'text',
                                                  'criteria': 'containing',
                                                  'value': 'lower',
                                                  'format': format2})

    # value low and high
    worksheet.set_column('E:I', 20, number_format)

    # Define our range for the color formatting
    color_range = "J2:J{}".format(nr + 1)
    worksheet.set_column(color_range, 20, total_percent_fmt)

    # Highlight the top 5 values in Green
    worksheet.conditional_format(color_range, {'type': 'data_bar'})

    # write total score rows
    total_fmt = workbook.add_format({'align': 'right', 'num_format': '0.00',
                                     'bold': True, 'bottom': 6})

    total_fmt_header = workbook.add_format({'align': 'right', 'num_format': '0.00',
                                            'bold': True, 'bottom': 6, 'bg_color': '#C6EFCE'})

    total_score = df['Raw Score'].sum()
    max_score = df['Max Score'].sum()
    performance = total_score / max_score

    worksheet.write_string(nr + 1, 5, "Max Score:", total_fmt)
    worksheet.write_string(nr + 1, 7, "Total Score:", total_fmt_header)

    # performance format
    performance_format = workbook.add_format(
        {'align': 'right', 'num_format': '0.0%', 'bold': True, 'bottom': 6, 'bg_color': '#C6EFCE'})

    cell_location = xl_rowcol_to_cell(nr + 1, 9)
    worksheet.write_number(cell_location, performance, performance_format)

    cell_location = xl_rowcol_to_cell(nr + 1, 6)
    # Get the range to use for the sum formula
    worksheet.write_number(cell_location, max_score, total_fmt)
    cell_location = xl_rowcol_to_cell(nr + 1, 8)
    worksheet.write_number(cell_location, total_score, total_fmt_header)

    # SAVE BANNER
    if banner_path is not None:
        options = {'x_scale': 0.87}
        worksheet.insert_image('A1', banner_path, options=options)

    # adding participant header
    # Create a format to use in the merged range.
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_size': 15,
    })

    # Merge 3 cells.
    worksheet.merge_range('A31:J31', report_header, merge_format)

    writer.save()


def download_wrapper(date, row, destination):
    row['date'] = date
    participant = row
    entry_folder = row['Name (First)'].capitalize() + ' ' + row['Name (Last)'].capitalize() + ' ' + str(row['Entry Id'])

    print(participant['Submit Plan'])
    try:
        uploaded_files = participant['Submit Plan'].split(',')
    except:
        txt = 'No files to download from : %s' % entry_folder
        logger.debug(txt)
        return

    entry_folder = row['Name (First)'].capitalize() + ' ' + row['Name (Last)'].capitalize() + ' ' + str(row['Entry Id'])
    entry_folder = entry_folder.replace('/', ' ')

    entry_path = osp.join(destination, entry_folder)

    if not osp.exists(entry_path):
        os.mkdir(entry_path)

        print(' Created entry directory: ', entry_folder)
    else:
        print("The directory already exists: %s" % entry_folder)

    # overide metadata
    participant.to_csv(osp.join(entry_path, 'metadata.csv'))

    for url in uploaded_files:
        try:
            _, file_name = osp.split(url)
            file_name_path = osp.join(entry_path, file_name)
            if not osp.exists(file_name_path):
                # Download the file from `url` and save it locally under `file_name`:
                with urllib.request.urlopen(url) as response, open(file_name_path, 'wb') as out_file:
                    data = response.read()  # a `bytes` object
                    out_file.write(data)
                    print('Saved file: %s inside: %s' % (file_name, entry_folder))
            else:
                print("The file already exists: %s" % file_name)
        except:
            txt = 'Error on downloading url: %s \n on folder: %s' % (url, entry_folder)
            logger.debug(txt)


class CompetitionPlans(object):
    def __init__(self, site_data_path):
        self.site_data = site_data_path
        self.df_raw = pd.read_csv(site_data_path, index_col=['Entry Date'])
        col = ['Name (First)',
               'Name (Last)',
               'Email (Enter Email)',
               'Country',
               'Plan Category',
               'Trial Plan Or Final Plan',
               'Treatment Planning System',
               'Technique',
               'Submit Plan',
               'Entry Id',
               'User Agent']
        self.df = self.df_raw[col].sort_index()

    def download_all(self, out_folder, pp=True):

        if pp:
            Parallel(n_jobs=-1, verbose=11)(
                delayed(download_wrapper)(date, row, out_folder) for date, row in self.df.iterrows())
        else:
            for date, row in self.df.iterrows():
                download_wrapper(date, row, out_folder)

    def generate_reports(self):
        pass


def test_download_all():
    destination = '/media/victor/TOURO Mobile/COMPETITION 2017/plans/dowload_website'
    f = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/submit-plan-2017-03-24.csv'

    cplans = CompetitionPlans(f)
    cplans.download_all(destination, pp=True)


def dvh_dose_stats(DVH):
    # generate dose stats per volume
    ordered_keys = list(DVH.keys())
    ordered_keys.sort(reverse=True)
    res = {}
    for key in ordered_keys:
        val = DVH[key]
        res[key] = {'min': val['min'],
                    'max': val['max'],
                    'mean': val['mean']}
    df = pd.DataFrame(res).T.round()

    return df


def read_config(file_path):
    # calculation options
    config = configparser.ConfigParser()
    config.read(file_path)
    calculation_options = dict()
    calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
    calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
    calculation_options['use_tps_structures'] = config.getboolean('DEFAULT', 'use_tps_structures')
    calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
    calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
    calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
    calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
    calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
    calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
    calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']
    return calculation_options


def parse_participant(root_path):
    files = [osp.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    filtered_files = OrderedDict()
    for f in files:
        try:
            obj = ScoringDicomParser(filename=f)
            rt_type = obj.GetSOPClassUID()
            if rt_type == 'rtdose':
                tmp = f.split(osp.sep)[-2].split()
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
            if rt_type == 'rtplan':
                tmp = f.split(osp.sep)[-2].split()
                name = tmp[0].split('-')[0]
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
        except:
            logger.debug('Error in file %s' % f)

    data = pd.DataFrame(filtered_files).T

    return data.reset_index()


def test_save_pdf():
    from pyplanscoring.core.scoring import Participant
    from pyplanscoring.core.dosimetric import read_scoring_criteria

    # plan data
    rd = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad Nobah/RD.1.2.246.352.71.7.584747638204.1758320.20170210154830.dcm'
    rp = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad Nobah/RP.1.2.246.352.71.5.584747638204.955801.20170210152428.dcm'
    rs = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad Nobah/RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm'
    dvh = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad Nobah/Test_ordering.dvh'

    # competition data
    folder = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring'
    path = osp.join(folder, 'Scoring Criteria.txt')
    constrains, scores, criteria = read_scoring_criteria(path)
    banner_path = osp.join(folder, '2017 Plan Comp Banner.jpg')

    # calculation options
    config = configparser.ConfigParser()
    config.read(osp.join(folder, 'PyPlanScoring.ini'))
    calculation_options = dict()
    calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
    calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
    calculation_options['use_tps_structures'] = config.getboolean('DEFAULT', 'use_tps_structures')
    calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
    calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
    calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
    calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
    calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
    calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
    calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']

    participant = Participant(rp, rs, rd, dvh_file=dvh, calculation_options=calculation_options)
    val = participant.eval_score(constrains, scores, criteria)
    report_df = participant.score_obj.get_report_df()

    out_report = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/competition/test.pdf'

    # Title.
    rep = CompetitionReportPDF(out_report, 'A3')
    rep.report(report_df, 'Test Competition Report', banner_path=banner_path)


def get_participant_data(participant_folder):
    metadata_path = osp.join(participant_folder, 'metadata.csv')
    if osp.exists(metadata_path):
        metadata = pd.read_csv(metadata_path,
                               header=None,
                               index_col=0,
                               parse_dates=True).to_dict('dict')[1]

        name = metadata['Name (First)'].capitalize() + ' ' + metadata['Name (Last)'].capitalize()

        files = [osp.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                 name.strip().endswith(('.dcm', '.DCM'))]

        filtered_files = OrderedDict()
        for f in files:
            try:
                obj = ScoringDicomParser(filename=f)
                rt_type = obj.GetSOPClassUID()
                if rt_type == 'rtdose':
                    participant_data = [name, rt_type, metadata['Plan Category'], metadata['Technique'],
                                        metadata['Treatment Planning System'], metadata['Trial Plan Or Final Plan']]
                    filtered_files[f] = participant_data
                if rt_type == 'rtplan':
                    participant_data = [name, rt_type, metadata['Plan Category'], metadata['Technique'],
                                        metadata['Treatment Planning System'], metadata['Trial Plan Or Final Plan']]

                    filtered_files[f] = participant_data
                if rt_type == 'rtss':
                    participant_data = [name, rt_type, metadata['Plan Category'], metadata['Technique'],
                                        metadata['Treatment Planning System'], metadata['Trial Plan Or Final Plan']]
                    filtered_files[f] = participant_data
            except:
                logger.debug('Error in file %s' % f)

        return True, pd.DataFrame(filtered_files).T, metadata

    else:
        logger.debug('There is no metadata.csv file inside %s' % participant_folder)
        return False, None


def get_dicom_data(participant_folder):
    files = [osp.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
             name.strip().endswith(('.dcm', '.DCM', '.dcm_', '.DCM_'))]

    filtered_files = OrderedDict()
    for f in files:
        try:
            obj = ScoringDicomParser(filename=f)
            rt_type = obj.GetSOPClassUID()
            if rt_type == 'rtdose':
                participant_data = [f, rt_type]
                filtered_files[f] = participant_data
            if rt_type == 'rtplan':
                participant_data = [f, rt_type]

                filtered_files[f] = participant_data
            if rt_type == 'rtss':
                participant_data = [f, rt_type]
                filtered_files[f] = participant_data
        except:
            logger.debug('Error in file %s' % f)

    return pd.DataFrame(filtered_files).T


class CompetitionReports(object):
    def __init__(self, root_folder, app_folder=''):
        self.root_folder = root_folder
        self.batch_data = []
        criteria_path = osp.join(app_folder, 'Scoring Criteria.txt')
        self.constrains, self.scores, self.criteria = read_scoring_criteria(criteria_path)
        self.banner_path = osp.join(app_folder, '2017 Plan Comp Banner.jpg')
        self.calculation_options = read_config(osp.join(app_folder, 'PyPlanScoring.ini'))
        self.app_folder = app_folder
        self.ref_plan_file = osp.join(app_folder, 'RP.1.2.246.352.71.5.584747638204.955801.20170210152428.dcm')
        self.ref_struc_file = osp.join(app_folder, 'RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm')
        self.set_participant_folder()

    def set_participant_folder(self):
        for folder in os.listdir(self.root_folder):
            participant_folder = osp.join(self.root_folder, folder)
            self.batch_data.append(participant_folder)

    def check_xio_version(self, dicom_data, f):
        try:
            rp = dicom_data.reset_index().set_index(1).ix['rtplan']['index']
            obj = ScoringDicomParser(filename=rp)
            tps_info = obj.get_tps_data()
            print(tps_info)

            if tps_info['Manufacturer'] == 'CPT2.0 Medical Software':
                try:
                    rs = dicom_data.reset_index().set_index(1).ix['rtss']['index']
                    return True, rs, dicom_data
                except:

                    return False, None, dicom_data

            if tps_info['TPS'].lower() == 'xio':

                if int(tps_info['SoftwareVersions'][0]) < 5:
                    try:
                        rs = dicom_data.reset_index().set_index(1).ix['rtss']['index']
                        return True, rs, dicom_data
                    except:

                        return False, None, dicom_data
                else:

                    return False, None, dicom_data
            else:
                return False, None, dicom_data
        except:
            # f, name = osp.split(self.ref_struc_file)
            logger.exception('Error on cheking Xio less than 4.00 on folder: %s' % f)
            return False, None, dicom_data

    def participant_report(self, folder_path):

        # test parse participant folder
        flag_folder, folder_data, metadata = get_participant_data(folder_path)

        # if not flag_folder:
        #     return None

        # check data consistency
        flag, truth, dicom_data = check_required_files(folder_data, folder_path)

        if flag and truth['rtdose']:

            # Set report header
            # todo sanitize name strings
            name = folder_data[0][0].replace('/', ' ')
            # plan_type = folder_data[2][0]
            # tech = folder_data[3][0]
            # tps_name = folder_data[4][0]
            # report_header = name + ' - ' + plan_type + ' - ' + tech + ' - ' + tps_name

            report_header = self.get_report_title(metadata)

            print('-------------')
            print(report_header)
            print('-------------')
            # Participant Data
            print('files:', dicom_data)
            # Check if there is XIO less than 4.00
            test_xio, rs, dicom_data = self.check_xio_version(dicom_data, folder_path)

            # Use ref rp file only to generate reports
            if not test_xio:
                rs = self.ref_struc_file

            rp = self.ref_plan_file
            rd_files = dicom_data.reset_index().set_index(1).ix['rtdose']['index']

            if not isinstance(rd_files, str):
                for rd_file in rd_files:
                    print('Calculating %s dose file' % rd_file)
                    try:
                        _, dose_file_name = osp.split(rd_file)
                        report_name = name + '_' + dose_file_name[:-5] + '_plan_report.pdf'
                        out_report = osp.join(folder_path, report_name)
                        pname = name + '_' + dose_file_name[:-5]
                        report_xls = name + '_' + dose_file_name[:-5] + '_plan_report.xlsx'
                        report_xls_path = osp.join(folder_path, report_xls)
                        if not osp.exists(out_report):
                            participant = Participant(rp, rs, rd_file,
                                                      calculation_options=self.calculation_options)
                            participant.set_participant_data(pname)
                            val = participant.eval_score(self.constrains, self.scores, self.criteria)
                            participant.save_score(report_xls_path, self.banner_path, report_header)
                            report_df = participant.score_obj.get_report_df()

                            # Title.
                            rep = CompetitionReportPDF(out_report, 'A3')
                            rep.report(report_df, report_header, banner_path=self.banner_path)
                            print('PDF report saved at: %s ' % out_report)
                            print('XLSX report saved at %s' % report_xls_path)


                    except:
                        logger.debug('Error in file: %s ' % rd_file)
                        logger.debug('Error in file: %s ' % rp)

            else:
                try:
                    _, dose_file_name = osp.split(rd_files)
                    report_name = name + '_' + dose_file_name[:-5] + '_plan_report.pdf'
                    report_xls = name + '_' + dose_file_name[:-5] + '_plan_report.xlsx'
                    report_xls_path = osp.join(folder_path, report_xls)
                    out_report = osp.join(folder_path, report_name)
                    pname = name + '_' + dose_file_name[:-5]
                    if not osp.exists(out_report):
                        participant = Participant(rp, rs, rd_files,
                                                  calculation_options=self.calculation_options)
                        participant.set_participant_data(pname)
                        val = participant.eval_score(self.constrains, self.scores, self.criteria)
                        participant.save_score(report_xls_path, self.banner_path, report_header)

                        report_df = participant.score_obj.get_report_df()

                        # Title.
                        rep = CompetitionReportPDF(out_report, 'A3')
                        rep.report(report_df, report_header, banner_path=self.banner_path)
                        print('PDF report saved at: %s ' % out_report)
                        print('XLSX report saved at %s' % report_xls_path)
                except:
                    logger.debug('Error in file: %s ' % rd_files)
                    logger.debug('Error in file: %s ' % rp)
        else:
            logger.debug("Missing dicom data: %s" % str(truth))

    def participant_dvh(self, folder_path):
        files_data = get_dicom_data(participant_folder=folder_path)
        rd = files_data.reset_index().set_index(1).ix['rtdose']['index']
        rp = files_data.reset_index().set_index(1).ix['rtplan']['index']
        rs = files_data.reset_index().set_index(1).ix['rtss']['index']
        p, filename = os.path.split(rd)
        participant = Participant(rp, rs, rd, calculation_options=self.calculation_options)
        participant.set_participant_data(filename[:-4])
        participant.eval_score(self.constrains, self.scores, self.criteria)

    def generate_dvhs(self):
        i = 0
        for p in self.batch_data:
            self.participant_dvh(p)
            print('DVH generation ITERATION: %i' % i)
            print('folder: ', p)
            i += 1

    def generate_reports(self):
        i = 0
        for p in self.batch_data:
            self.participant_report(folder_path=p)
            print('ITERATION: %i' % i)
            i += 1

    def delete_reports(self):
        files = [osp.join(root, name) for root, dirs, files in os.walk(self.root_folder) for name in files if
                 name.strip().endswith(('.PDF', '.pdf', '.xlsx'))]

        for rep in files:
            os.remove(rep)

    def spreadsheet_pdf(self, root_folder):
        for folder in os.listdir(root_folder):
            participant_folder = osp.join(root_folder, folder)

            files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name in
                     files if name.strip().endswith('.xlsx')]

            csv_file = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name
                        in files if name.strip().endswith('.csv')]

            metadata = pd.read_csv(csv_file[0], header=None, index_col=0, parse_dates=True).to_dict('dict')[1]

            report_header = self.get_report_title(metadata)

            for xls_file in files:
                file_path, ext = osp.splitext(xls_file)
                out_report = file_path + '.pdf'
                report_df = pd.read_excel(xls_file, header=31).dropna()
                rep = CompetitionReportPDF(out_report, 'A3')
                rep.report(report_df, report_header, banner_path=self.banner_path)

    def set_xls_titles(self, root_folder):

        for folder in os.listdir(root_folder):
            participant_folder = osp.join(root_folder, folder)

            files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for
                     name in
                     files if name.strip().endswith('.xlsx')]

            csv_file = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for
                        name
                        in files if name.strip().endswith('.csv')]

            metadata = pd.read_csv(csv_file[0], header=None, index_col=0, parse_dates=True).to_dict('dict')[1]

            report_header = self.get_report_title(metadata)

            for xls_file in files:
                report_df = pd.read_excel(xls_file, header=31).dropna()
                save_formatted_report(report_df, xls_file, self.banner_path, report_header=report_header)

    @staticmethod
    def get_report_title(metadata):
        name = metadata['Name (First)'].capitalize() + ' ' + metadata['Name (Last)'].capitalize()
        name = name.replace('/', ' ')
        plan_type = metadata['Plan Category']
        tech = metadata['Technique']
        tps_name = metadata['Treatment Planning System']
        trial_final = metadata['Trial Plan Or Final Plan']
        try:
            datei = datetime.datetime.strptime(metadata['date'], "%Y-%m-%d  %H:%M:%S")
            date = datetime.datetime.strftime(datei, "%b %d %Y (%H:%M:%S KSA Time)")
            report_header = '%s, %s, %s, %s, %s, %s' % (name, tps_name, tech, plan_type, trial_final, date)
        except:
            logger.debug('failed to parse dates from %s %s' % (name, metadata['Entry Id']))
            report_header = '%s, %s, %s, %s, %s' % (name, tps_name, tech, plan_type, trial_final)

        return report_header


def check_required_files(dicom_data, folder_path):
    """
        Check if there is at least one RP, RS, RD file inside Participant folder data
        
    :param dicom_data: Participant DataFrame
    :return: Truth dictionary
    """
    truth = {}
    dcm_types = ['rtss', 'rtdose', 'rtplan']
    try:
        for t in dcm_types:
            val = t in dicom_data[1].values
            truth[t] = val
            if not val:
                logger.debug('%s file not found at: %s' % (t, folder_path))
    except:
        logger.debug('Dicom data not found at: %s' % folder_path)

    try:
        rd = dicom_data.reset_index().set_index(1).ix['rtdose']['index']
        # rs = dicom_data.reset_index().set_index(1).ix['rtss']['index']
        # rp = dicom_data.reset_index().set_index(1).ix['rtplan']['index']
        return True, truth, dicom_data
    except:
        logger.debug('RDOSE Dicom data not found at: %s' % folder_path)

        return False, truth, None


def test_update_reports():
    # download data
    root_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/plans/dowload_website'
    app_folder = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring'
    comp = CompetitionReports(root_folder, app_folder)
    comp.set_participant_folder()
    # comp.delete_reports()
    comp.generate_reports()

    reports_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/plans/reports_pdf'
    for folder in os.listdir(root_folder):
        participant_report_folder = osp.join(reports_folder, folder)
        participant_folder = osp.join(root_folder, folder)
        if not osp.exists(participant_report_folder):
            os.mkdir(participant_report_folder)

        files = [osp.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                 name.strip().endswith(('.PDF', '.pdf', '.csv', '.xlsx'))]

        for f in files:
            shutil.copy(f, participant_report_folder)


class SendReports(object):
    def __init__(self, login='', password=''):
        self.gmail_user = login
        self.gmail_pwd = password
        self.report_data = {}

    def mail(self, to, subject, text, attach=None):
        """
            Edit:Gmail imposes a number of restrictions on people sending email with SMTP.
            Free accounts are limited to 500 emails per day and are rate-limited to about  20 emails per second.
        :param to: dest mail
        :param subject: Email-subject
        :param text: Text body
        :param attach: file path to the attachment
        """
        time.sleep(0.1)
        msg = MIMEMultipart()
        msg['From'] = self.gmail_user
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(text))
        if attach:
            for at in attach:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(open(attach, 'rb').read())
                Encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment; filename="%s"' % osp.basename(attach))
                msg.attach(part)
        mailServer = smtplib.SMTP("smtp.gmail.com", 587)
        mailServer.ehlo()
        mailServer.starttls()
        mailServer.ehlo()
        mailServer.login(self.gmail_user, self.gmail_pwd)
        mailServer.sendmail(self.gmail_user, to, msg.as_string())
        mailServer.close()

    def send_report_data(self, reports_folder, subject):

        for folder in os.listdir(reports_folder):
            participant_folder = osp.join(reports_folder, folder)

            files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name in
                     files if
                     name.strip().endswith('.pdf')]

            csv_file = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name
                        in files if name.strip().endswith('.csv')]

            metadata = pd.read_csv(csv_file[0], header=None, index_col=0, parse_dates=True).to_dict('dict')[1]

            name = metadata['Name (First)'].capitalize() + ' ' + metadata['Name (Last)'].capitalize()
            name = name.replace('/', ' ')
            dest = metadata['Email (Enter Email)']

            text = self.make_email_body(name)

            # send an email to each participant
            self.mail(dest, subject, text, files)

    def make_email_body(self, name):
        body = "Dear %s,\n" \
               "We have evaluated your submitted plan.\n" \
               "Kindly find your score sheet attached to this email.\n" \
               "Good luck\n" \
               "Regards\n" \
               "Radiation Knowledge Team\n" % name

        return body


def join_entries(entries_folder, csv_files):
    df_temp = []
    for f in csv_files:
        df_temp.append(pd.read_csv(f, index_col=14, parse_dates=False))

    df = pd.concat(df_temp, join='inner')
    df = df.drop_duplicates()

    df.sort_index(ascending=False).to_csv(osp.join(entries_folder, 'entries.csv'))

    return df


def parse_participant_plan_data(participant_folder):
    # TODO add default values and error handling
    files_data = get_dicom_data(participant_folder)

    tmp = files_data.reset_index().set_index(1)

    rd = tmp.ix['rtdose']['index'] if 'rtdose' in tmp.index else ''
    rp = tmp.ix['rtplan']['index'] if 'rtplan' in tmp.index else ''
    rs = tmp.ix['rtss']['index'] if 'rtss' in tmp.index else ''

    plan_dict = dict()

    try:
        obj = ScoringDicomParser(filename=rp)
        _, filename = osp.split(rp)
        plan_dict = obj.GetPlan()
        plan_dict['plan_filename'] = filename
        # dose filename
        _, dfilename = osp.split(rd)
        plan_dict['dose_filename'] = dfilename
        # struc filename
        # dose filename
        _, sfilename = osp.split(rs)
        plan_dict['structure_filename'] = sfilename

    except:
        logger.exception('Error in plan file %s' % rp)

    # grap plan info data
    plan_info = OrderedDict()
    plan_info['Plan file'] = plan_dict['plan_filename']
    plan_info['Structure file'] = plan_dict['structure_filename']
    plan_info['Dose file'] = plan_dict['dose_filename']
    plan_info['Number of beams/arcs'] = str(len(plan_dict['beams']))
    plan_info['Prescribed dose [cGy]'] = str(plan_dict['rxdose'])
    plan_info['Total MU'] = str(round(plan_dict['Plan_MU']))
    plan_info['Number of isocenters'] = str(plan_dict['n_isocenters'])

    return plan_info


def parse_plan_pp(root_folder, folder):
    participant_folder = osp.join(root_folder, folder)
    print('-----------')
    print('Folder: %s' % folder)
    files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name in
             files if name.strip().endswith('.dcm')]

    plan_files = []
    plan_data = []
    for f in files:
        print('file: %s' % f)
        # try:
        obj = ScoringDicomParser(filename=f)
        rt_type = obj.GetSOPClassUID()
        if rt_type == 'rtplan':
            plan_files.append(f)
            plan_data.append(obj.GetPlan())
            # except:
            #     logger.exception('Error in file %s' % f)

    return folder, plan_files, plan_data


class GenerateReports(object):
    def __init__(self, app_folder, root_folder, reports_folder, error_folder):
        self.app_folder = app_folder
        self.root_folder = root_folder
        self.reports_folder = reports_folder
        self.error_folder = error_folder
        self.reports = CompetitionReports(root_folder=root_folder, app_folder=app_folder)
        self.plans = None

    def download_plans(self, csv_path):
        self.plans = CompetitionPlans(csv_path)
        self.plans.download_all(self.root_folder)

    def calc_report(self, clean_data=False):
        if clean_data:
            self.reports.delete_reports()

        self.reports.generate_reports()

    def participant_report(self, participant_folder):
        self.reports.participant_report(participant_folder)

    def save_reports(self):
        for folder in os.listdir(self.root_folder):
            participant_report_folder = osp.join(self.reports_folder, folder)
            participant_folder = osp.join(self.root_folder, folder)
            if not osp.exists(participant_report_folder):
                os.mkdir(participant_report_folder)

            files = [osp.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                     name.strip().endswith(('.PDF', '.pdf', '.csv', '.xlsx'))]

            for f in files:
                _, fname = osp.split(f)
                shutil.copy(f, participant_report_folder)
                print('Saved file %s at %s' % (fname, participant_report_folder))

    def move_error_plans(self):
        for folder in os.listdir(self.root_folder):
            participant_folder = osp.join(self.root_folder, folder)
            files = [osp.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                     name.strip().endswith(('.pdf', '.xlsx'))]

            if not files:
                logger.debug('No report at: %s' % participant_folder)
                shutil.move(participant_folder, self.error_folder)

    def xls2pdf(self):
        self.reports.spreadsheet_pdf(self.reports_folder)

    def fix_xls_header(self):
        self.reports.set_xls_titles(self.reports_folder)


def get_calculated_dvh_data(participant_folder):
    # TODO add exeption to no DVH data
    files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name in
             files if name.strip().endswith('.dvh')]
    dvh = load(files[0])

    return dvh


def get_xlsx_report_data(participant_folder, header_index=17):
    files = [osp.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for
             name in
             files if name.strip().endswith('.xlsx')]

    for xls_file in files:
        report_df = pd.read_excel(xls_file, header=header_index).dropna()
        report_header = pd.read_excel(xls_file, header=header_index - 2).dropna().columns[0]
        return report_df, report_header


def save_dvh_report(DVH, dest_path):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # A3 landascape
    fig.set_size_inches(16.53, 11.69)
    ordered_keys = list(DVH.keys())
    ordered_keys.sort(reverse=True)
    major_y_ticks = range(0, 101, 20)
    minor_y_ticks = range(0, 101, 5)
    major_x_ticks = range(0, 8020, 1000)
    minor_x_ticks = range(0, 8020, 500)

    ax.set_xticks(major_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)
    ax.set_xlim([0, 8020])

    ax.grid(which='both')

    for key in ordered_keys:
        val = DVH[key]
        y = val['data'] / val['data'][0] * 100
        x = val['dose_axis']
        ax.plot(x, y, label=key, linewidth=1.0)
        ax.legend(loc=7, borderaxespad=-11)

        ax.set_ylabel('volume (%)')
        ax.set_xlabel('Dose (cGy)')

        apendix = 'PyPlanScoring - Calculated DVH - Voxel Size [mm]: (0.2, 0.2, 0.2) '
        ax.set_title(apendix)
        # data.append(Paragraph(apendix, styles['Participant Header']))

    fig.savefig(dest_path, format='png', dpi=100)
    plt.close('all')


def final_report_snippet():
    # set final report
    participant_folder = r'D:\Dropbox\Plan_Competition_Project\competition_2017\plans\TO VICTOR\Abdul Qadir Jangda - Eclipse - IMRT - 23 MARCH FINAL - 50.4'
    pdatga = get_participant_data(participant_folder)

    # plan data
    plan_data_report = parse_participant_plan_data(participant_folder)

    # getting DVH file and data
    DVH = get_calculated_dvh_data(participant_folder)
    dvh_stats = dvh_dose_stats(DVH)

    # saving dvh figure
    out_dvh_img = osp.join(participant_folder, 'report_dvh.png')
    save_dvh_report(DVH, out_dvh_img)
    # plt.show()

    # # GET XLSX REPORT
    report_df, report_header = get_xlsx_report_data(participant_folder)

    # save PDF report
    banner_path = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\2017 Plan Comp Banner.jpg'

    out_report = osp.join(participant_folder, 'Victor_FINAL_report.pdf')
    rep = FinalReportPDF(out_report, 'A3')
    rep.final_report(report_df, dvh_stats, report_header, banner_path=banner_path, dvh_path=out_dvh_img)


class FinalReportGenerator(object):
    def __init__(self, root_folder, app_folder):
        self.root_folder = root_folder
        self.app_folder = app_folder
        self.banner_path = osp.join(app_folder, '2017 Plan Comp Banner.jpg')
        self.batch_data = []

    def set_participant_folder(self):
        for folder in os.listdir(self.root_folder):
            participant_folder = osp.join(self.root_folder, folder)
            self.batch_data.append(participant_folder)

    def gen_final_report(self, participant_folder, header_index):
        # plan data
        plan_info = parse_participant_plan_data(participant_folder)

        # getting DVH file and data
        DVH = get_calculated_dvh_data(participant_folder)
        dvh_curves = DVH['DVH']
        part_name = DVH['participant']

        # saving dvh figure

        dvh_stats = dvh_dose_stats(dvh_curves)
        # saving dvh figure
        out_dvh_img = osp.join(participant_folder, 'report_dvh.png')
        save_dvh_report(dvh_curves, out_dvh_img)

        # getting data from xlsx report
        report_df, report_header = get_xlsx_report_data(participant_folder, header_index)

        # out report filename
        out_report = osp.join(participant_folder, part_name + '_final_report.pdf')

        # saving PDF final report
        rep = FinalReportPDF(out_report, 'A3')
        rep.final_report(report_df, dvh_stats, report_header, plan_info, banner_path=self.banner_path,
                         dvh_path=out_dvh_img)

    def batch_final_report(self):
        i = 0
        errors = []
        for p in self.batch_data:
            _, pname = osp.split(p)
            print('Participant %s final report ITERATION: %i' % (pname, i))
            try:
                self.gen_final_report(p, header_index=17)
            except:
                errors.append(p)
                print('Error in folder', p)
                print('try new header index')
                self.gen_final_report(p, header_index=31)
            finally:
                print('no report generated for folder ', p)
                errors.append(p)
            i += 1
        return errors

    def batch_final_pp(self):

        Parallel(n_jobs=-1, verbose=11)(delayed(self.report_wrapper)(p) for p in self.batch_data)

    def report_wrapper(self, p):
        _, pname = osp.split(p)
        print('Participant %s final' % pname)
        try:
            self.gen_final_report(p, header_index=17)
        except:
            print('Error in folder', p)
            print('try new header index')
            self.gen_final_report(p, header_index=31)


def download_missing_dicom():
    import os
    import pandas as pd
    from pyplanscoring.core.dicomparser import ScoringDicomParser
    import urllib
    import urllib.request
    import logging

    logger = logging.getLogger('utils.py')

    logging.basicConfig(filename='Generate_reports.log', level=logging.DEBUG)

    def get_dicom_files(participant_folder):
        return [os.path.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                name.strip().endswith(('.dcm', '.DCM'))]

    def check_dicom_files(participant_folder):
        files = get_dicom_files(participant_folder)
        filtered_files = []
        for f in files:
            try:
                obj = ScoringDicomParser(filename=f)
                rt_type = obj.GetSOPClassUID()
                if rt_type == 'rtdose' or rt_type == 'rtplan' or rt_type == 'rtdose':
                    filtered_files.append(f)
            except:
                logger.debug('Error in file %s' % f)

        return filtered_files

    def download_metadata(participant, participant_folder):
        print(participant['Submit Plan'])
        uploaded_files = participant['Submit Plan'].split(',')
        for url in uploaded_files:
            try:
                _, file_name = os.path.split(url)
                file_name_path = os.path.join(participant_folder, file_name)
                # Download the file from `url` and save it locally under `file_name`:
                with urllib.request.urlopen(url) as response, open(file_name_path, 'wb') as out_file:
                    data = response.read()  # a `bytes` object
                    out_file.write(data)
                    print('Saved file: %s inside: %s' % (file_name, participant_folder))

            except:
                txt = 'Error on downloading url: %s \n on folder: %s' % (url, participant_folder)
                logger.debug(txt)

    root_folder = r'D:\Final_Plans\ECPLIPSE_VMAT'
    participant = {}
    for folder in os.listdir(root_folder):
        participant_folder = os.path.join(root_folder, folder)
        metadata_path = os.path.join(participant_folder, 'metadata.csv')
        if os.path.exists(metadata_path):
            # check there is at least RP and RD files
            files = check_dicom_files(participant_folder)
            if len(files) < 2:
                metadata = pd.read_csv(metadata_path, header=None, index_col=0, parse_dates=True).to_dict('dict')[1]
                download_metadata(metadata, participant_folder)


def copy_missing_dicom():
    import os
    import shutil

    import pandas as pd
    import re

    from pyplanscoring.core.dicomparser import ScoringDicomParser
    import urllib
    import urllib.request
    import logging

    logger = logging.getLogger('utils.py')

    logging.basicConfig(filename='Generate_reports.log', level=logging.DEBUG)

    def get_dicom_files(participant_folder):
        return [os.path.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                name.strip().endswith(('.dcm', '.DCM', '.dcm_', '.DCM_'))]

    def get_dvh_file(participant_folder):
        return [os.path.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                name.strip().endswith('.dvh')]

    def check_dicom_files(participant_folder):
        files = get_dicom_files(participant_folder)
        filtered_files = []
        for f in files:
            try:
                obj = ScoringDicomParser(filename=f)
                rt_type = obj.GetSOPClassUID()
                if rt_type == 'rtdose' or rt_type == 'rtplan' or rt_type == 'rtdose':
                    filtered_files.append(f)
            except:
                logger.debug('Error in file %s' % f)

        return filtered_files

    def download_metadata(participant, participant_folder):
        print(participant['Submit Plan'])
        uploaded_files = participant['Submit Plan'].split(',')
        for url in uploaded_files:
            try:
                _, file_name = os.path.split(url)
                file_name_path = os.path.join(participant_folder, file_name)
                # Download the file from `url` and save it locally under `file_name`:
                with urllib.request.urlopen(url) as response, open(file_name_path, 'wb') as out_file:
                    data = response.read()  # a `bytes` object
                    out_file.write(data)
                    print('Saved file: %s inside: %s' % (file_name, participant_folder))

            except:
                txt = 'Error on downloading url: %s \n on folder: %s' % (url, participant_folder)
                logger.debug(txt)

    # Script to copy missing dicom data

    search_folder = r'I:\COMPETITION 2017\submited_plans_15_april\submited_plans_15_april\plans'
    root_folder = r'I:\COMPETITION 2017\final_plans\Other\VICTOR M-Z'
    participant = {}
    for folder in os.listdir(root_folder):
        participant_folder = os.path.join(root_folder, folder)
        metadata_path = os.path.join(participant_folder, 'metadata.csv')
        if os.path.exists(metadata_path):
            # check there is at least RP and RD files
            files = check_dicom_files(participant_folder)
            dvh_files = get_dvh_file(participant_folder)
            if len(files) < 2:
                print(metadata_path)
                metadata = pd.read_csv(metadata_path, header=None, index_col=0, parse_dates=True).to_dict('dict')[1]

                # search at database
                for f in os.listdir(search_folder):
                    if re.findall(metadata['Entry Id'], f):
                        db_folder = os.path.join(search_folder, f)
                        dcm_files = check_dicom_files(db_folder)
                        # copy files to destination
                        for dcm in dcm_files:
                            shutil.copy(dcm, participant_folder)
                            print('Copied: %s to %s' % (dcm, participant_folder))

            # check dvh files
            if not dvh_files:
                # print(metadata_path)
                metadata = pd.read_csv(metadata_path, header=None, index_col=0, parse_dates=True).to_dict('dict')[1]

                # search at database
                for f in os.listdir(search_folder):
                    if re.findall(metadata['Entry Id'], f):
                        db_folder = os.path.join(search_folder, f)
                        dvh_files = get_dvh_file(db_folder)
                        # copy files to destination
                        for dvh in dvh_files:
                            shutil.copy(dvh, participant_folder)
                            print('Copied: %s to %s' % (dvh, participant_folder))


if __name__ == '__main__':
    root_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/ECPLIPSE_VMAT'
    app_folder = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring'
    frg = FinalReportGenerator(root_folder=root_folder, app_folder=app_folder)
    frg.set_participant_folder()
    errors = frg.batch_final_report()
    # frg.batch_final_pp()
