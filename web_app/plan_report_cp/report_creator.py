import sys

import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell

from pyplanscoring.core.dicomparser import ScoringDicomParser
from pyplanscoring.core.dosimetric import read_scoring_criteria
from pyplanscoring.core.scoring import Participant

if sys.version[0] == '2':
    import cStringIO

    output = cStringIO.StringIO()
else:
    # python3.4
    from io import BytesIO

    output = BytesIO()


class PlanReportCreator(object):
    def __init__(self):
        self.results_df = None
        self.rp_file = ''
        self.rs_file = ''
        self.rd_file = ''
        self.rp_dcm = None
        self.rs_dcm = None
        self.rd_dcm = None

    # Load a struct file only. Code is similar to loadFolder()
    def loadStruct(self, filename):
        obj = ScoringDicomParser(filename=filename)
        if obj.GetSOPClassUID() != 'rtss':
            return "Not an RTStruct file"
        else:
            self.rs_file = filename
            self.rs_dcm = obj
            return 0

    # Load a dose file only. Code is similar to loadFolder()
    def loadDose(self, filename):
        obj = ScoringDicomParser(filename=filename)
        if obj.GetSOPClassUID() != 'rtdose':
            return "Not an RTDose file"
        else:
            self.rd_file = filename
            self.rd_dcm = obj
            return 0

    def load_plan(self, filename):
        # Load a dose file only. Code is similar to loadFolder()
        obj = ScoringDicomParser(filename=filename)
        if obj.GetSOPClassUID() != 'rtplan':
            return "Not an RTPlan file"
        else:
            self.rp_file = filename
            self.rp_dcm = obj
            return 0

    def makeReport(self, criteria_txt, banner_path='', report_header=''):
        constrains, scores, criteria = read_scoring_criteria(criteria_txt)
        # Set calculation options
        calculation_options = dict()
        calculation_options['end_cap'] = 0.2
        calculation_options['use_tps_dvh'] = False
        calculation_options['up_sampling'] = True
        calculation_options['maximum_upsampled_volume_cc'] = 100.0
        calculation_options['voxel_size'] = 0.2
        calculation_options['num_cores'] = 8
        calculation_options['save_dvh_figure'] = False
        calculation_options['save_dvh_data'] = False
        calculation_options['mp_backend'] = 'threading'

        print('------------- Calculating DVH and score --------------')
        participant = Participant(self.rp_file, self.rs_file, self.rd_file, calculation_options=calculation_options)
        participant.set_participant_data(report_header)
        val = participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria)

        df_report = participant.get_score_report()
        print('Plan Score: %1.2f' % val)
        return self.save_formatted_report(df_report, banner_path=banner_path, report_header=report_header)

    @staticmethod
    def save_formatted_report(df, banner_path=None, report_header=''):

        """
            Save an formated report using pandas and xlsxwriter and
        :param df: Results dataframe
        :param out_file: filename path
        :param banner_path: banner path

        """
        start_row = 31
        number_rows = len(df.index)
        # Use the StringIO object as the filehandle.
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        # Write the data frame to the StringIO object.
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

        # return xlsx_data to response value
        xlsx_data = output.getvalue()
        return xlsx_data


if __name__ == "__main__":
    pass
