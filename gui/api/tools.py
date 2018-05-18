import os
from typing import Dict, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xlsxwriter.utility import xl_rowcol_to_cell

from core.dicom_reader import PyDicomParser


def read_eclipse_dvh(file_path: str) -> Dict[str, np.ndarray]:
    """
        Reads eclipse DVH data and return a dictionary
    :param file_path: path_to_txt_file
    :return:
    """
    with open(file_path, 'r') as f:
        txt = f.readlines()

    flag_dvh = False
    structure_header = False
    sname = ''
    dvhs = {}
    for l in txt:
        # getting structure name
        if re.match('Structure', l):
            tmp = l.split()
            sname = ' '.join(tmp[1:])
            structure_header = True
            flag_dvh = False
            continue

        if re.findall('Structure Volume', l):
            # begin data
            structure_header = False
            flag_dvh = True
            structure_dvh = []
            continue

        if not structure_header and flag_dvh:
            # getting decimal data
            if re.findall(r"[-+]?\d*\.*\d+", l):
                structure_dvh.append(l.split())
            else:
                flag_dvh = False
                data = np.array(structure_dvh, dtype=float)
                # to Gy
                data[:, 0] = data[:, 0] / 100.0
                dvhs[sname] = data

    return dvhs


def read_slicer_dvh(file_path):
    df_slicer = pd.read_csv(file_path).dropna(axis=1)
    values_axis = df_slicer.iloc[:, 1::2]
    dose_axis = df_slicer.iloc[:, 0]
    columns = values_axis.columns
    volumes = np.array([re.findall('(\d+(?:\.\d+))', name)[0] for name in columns], dtype=float)
    values_axis = values_axis * volumes / 100
    values_axis['dose_axis'] = dose_axis
    slicer_dvh = values_axis.set_index('dose_axis')
    return slicer_dvh


def plot_dvh(dvh_calc, title):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])

    plt.plot(x_calc, dvh_calc['data'], label='PyPlanScoring')
    # plt.xlim([x_calc.min(), x_calc.max()])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)


def get_dicom_data(root_path: str) -> Dict[str, List[str]]:
    """
        Provide all participant required files (RP,RS an RD DICOM FILES)
    :param root_path: participant folder
    :return: Pandas DataFrame containing path to files
    """
    files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    filtered_files = []
    for f in files:
        obj = PyDicomParser(filename=f)
        rt_type = obj.GetSOPClassUID()
        # fix halcyon SOP class UI
        if rt_type is None:
            rt_type = obj.ds.Modality.lower()

        if rt_type in ['rtdose', 'rtplan', 'rtss']:
            filtered_files.append([rt_type, f])

    # filter rtdose
    rd_files = [d[1] for d in filtered_files if d[0] == 'rtdose']
    rp_files = [d[1] for d in filtered_files if d[0] == 'rtplan']
    rs_files = [d[1] for d in filtered_files if d[0] == 'rtss']

    dcm_files = {'rtdose': rd_files, 'rtplan': rp_files, 'rtss': rs_files}

    return dcm_files


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
