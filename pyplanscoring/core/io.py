import bz2
import json
import os
import pickle

from pydicom.valuerep import IS
from xlsxwriter.utility import xl_rowcol_to_cell

from core.dicom_reader import PyDicomParser
import pandas as pd


def load(filename):
    """
        Loads a Calibration Object into a file using gzip and Pickle
    :param filename: Calibration filemane *.fco
    :return: object
    """
    with bz2.BZ2File(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save(obj, filename, protocol=-1):
    """
        Saves  Object into a file using gzip and Pickle
    :param obj: Calibration Object
    :param filename: Filename *.fco
    :param protocol: cPickle protocol
    """
    # TODO check if compression is necessary
    with bz2.BZ2File(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def save_dvh_json(dvh_data_dict, file_path_name):
    """
        Helper function to save dvh_data into JSON file
    :param dvh_data_dict:
    :param file_path_name:
    """

    with open(file_path_name, 'w', encoding='utf-8') as json_file:
        json.dump(dvh_data_dict,
                  json_file,
                  ensure_ascii=False)


def load_dvh_json(file_path_name):
    """

    :param file_path_name:
    :return:
    """

    with open(file_path_name, 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        # add pydicom key type (int)
        json_dict = {IS(k): v for k, v in json_dict.items()}
        return json_dict


def normalize_data(dvh_data_dict):
    pass


class IOHandler:

    def __init__(self, dvh_data_dict=None, header_info=None):
        """
            Class to encapsulate IO methods for DVH data storage
            It receives a PyPlanScoring DVH data dictionary
        :param dvh_data_dict: PyPlanScoring DVH data dictionary
        """
        if dvh_data_dict is None:
            dvh_data_dict = {}
        self._header = None
        self._dvh_data = {}

        # setters
        self.dvh_data = dvh_data_dict
        self.header = header_info

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        # TODO define appropriate header
        self._header = value

    @property
    def dvh_data(self):
        return self._dvh_data

    @dvh_data.setter
    def dvh_data(self, value):
        # TODO add object checks
        self._dvh_data = dict(value)

    def to_dvh_file(self, file_path_name):
        """
            Save pickle *.dvh file
        :param file_path_name:
        """
        save(self.dvh_data, file_path_name)

    def read_dvh_file(self, file_path_name):
        """
            Loads pickle *.dvh file
        :param file_path_name:
        """
        self.dvh_data = load(file_path_name)

        return self.dvh_data

    def to_json_file(self, file_path_name):
        """
            Saves serialized dvh data into *.jdvh json file
        :param file_path_name:
        """
        save_dvh_json(self.dvh_data, file_path_name)

    def read_json_file(self, file_path_name):
        """
            Saves serialized dvh data into *.jdvh json file
        :param file_path_name:
        """
        self.dvh_data = load_dvh_json(file_path_name)
        return self.dvh_data


def get_participant_folder_data(root_path):
    """
        Provide all participant required files (RP,RS an RD DICOM FILES)
    :param root_path: participant folder
    :return: Pandas DataFrame containing path to files
    """
    files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    data_truth = []
    filtered_files = {'rtdose': False, 'rtplan': False, 'rtss': False}
    for f in files:
        obj = PyDicomParser(filename=f)
        rt_type = obj.GetSOPClassUID()
        # fix halcyon SOP class UI
        if rt_type is None:
            rt_type = obj.ds.Modality.lower()

        if rt_type in ['rtdose', 'rtplan', 'rtss']:
            filtered_files[rt_type] = f
            data_truth.append(True)

    missing_files = [key for key, value in filtered_files.items() if value is False]

    if len(data_truth) == 3 and not missing_files:
        return filtered_files, True
    else:
        return missing_files, False


def save_formatted_report(df, out_file, start_row=0, banner_path=None, report_header=''):
    """
        Save an formated report using pandas and xlsxwriter
    :param df: Results dataframe
    :param out_file: filename path
    :param banner_path: banner path
    """
    # start_row = 31
    # add performance

    df['Performance'] = df['Raw score'] / df['Score']

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
    total_percent_fmt = workbook.add_format({'align': 'right', 'num_format': '0.00%', 'bold': True})

    # # Total percent format
    percent_fmt = workbook.add_format({'align': 'right', 'num_format': '0.00%', 'bold': True})

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
                                                  'value': 'max',
                                                  'format': format1})

    # Highlight the bottom 5 values in Red
    worksheet.conditional_format(constrain_type, {'type': 'text',
                                                  'criteria': 'containing',
                                                  'value': 'min',
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

    total_score = df['Raw score'].sum()
    max_score = df['Score'].sum()
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
    # worksheet.merge_range('A31:J31', report_header, merge_format)

    # hide column A
    worksheet.set_column('A:A', None, None, {'hidden': True})

    perf = "D2:D{}".format(nr + 1)

    writer.save()
