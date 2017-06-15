import sys
from random import choice

from matplotlib import pyplot as plt
from reportlab.lib import colors, styles
from reportlab.lib.colors import HexColor, black, yellow
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter, A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable

if sys.version[0] == '2':
    import cStringIO

    output = cStringIO.StringIO()
else:
    # python3.4d
    from io import BytesIO

    output = BytesIO()


def stylesheet():
    styles = {'default': ParagraphStyle(
        name='default',
        fontName='Times-Roman',
        fontSize=12,
        leading=12,
        leftIndent=0,
        rightIndent=0,
        firstLineIndent=0,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
        bulletFontName='Times-Roman',
        bulletFontSize=10,
        bulletIndent=0,
        textColor=black,
        backColor=None,
        wordWrap=None,
        borderWidth=0,
        borderPadding=0,
        borderColor=None,
        borderRadius=None,
        allowWidows=1,
        allowOrphans=0,
        textTransform=None,  # 'uppercase' | 'lowercase' | None
        endDots=None,
        splitLongWords=1,
    )}
    styles['title'] = ParagraphStyle(
        'title',
        parent=styles['default'],
        fontName='Times-Bold',
        fontSize=16,
        leading=32,
        alignment=TA_CENTER,
        textColor=black,
    )
    styles['alert'] = ParagraphStyle(
        'alert',
        parent=styles['default'],
        leading=14,
        backColor=yellow,
        borderColor=black,
        borderWidth=1,
        borderPadding=5,
        borderRadius=2,
        spaceBefore=10,
        spaceAfter=10,
    )
    # add custom paragraph style
    styles['Participant Header'] = ParagraphStyle(
        name="Participant Header",
        fontSize=14,
        alignment=TA_CENTER,
        fontName='Times-Bold')

    styles["TableHeader"] = ParagraphStyle(
        name="TableHeader",
        fontSize=9,
        alignment=TA_CENTER,
        fontName='Times-Bold')

    styles["structure"] = ParagraphStyle(
        name="structure",
        fontSize=9,
        alignment=TA_LEFT,
        fontName='Times-bold')

    styles["Text"] = ParagraphStyle(
        name="Text",
        fontSize=9,
        alignment=TA_CENTER,
        fontName='Times')

    styles["upper"] = ParagraphStyle(
        name="upper",
        fontSize=9,
        alignment=TA_CENTER,
        fontName='Times',
        backColor=colors.lightcoral)

    styles["lower"] = ParagraphStyle(
        name="lower",
        fontSize=9,
        alignment=TA_CENTER,
        fontName='Times',
        backColor=colors.lightgreen)

    styles["number"] = ParagraphStyle(
        name="number",
        fontSize=9,
        alignment=TA_RIGHT,
        fontName='Times')

    styles["TextMax"] = ParagraphStyle(
        name="TextMax",
        fontSize=9,
        alignment=TA_RIGHT,
        fontName='Times-Bold')

    styles["Result number"] = ParagraphStyle(
        name="Result number",
        fontSize=9,
        alignment=TA_RIGHT,
        fontName='Times-Bold',
        backColor=colors.lightgreen)

    styles["Result"] = ParagraphStyle(
        name="Result",
        fontSize=9,
        alignment=TA_RIGHT,
        fontName='Times-bold')

    styles["number_dvh"] = ParagraphStyle(
        name="number_dvh",
        fontSize=12,
        alignment=TA_CENTER,
        fontName='Times')

    return styles


class CompetitionReportPDF(object):
    def __init__(self, buffer, pageSize='A4'):
        self.buffer = buffer
        # default format is A4
        if pageSize == 'A4':
            self.pageSize = A4
        elif pageSize == 'Letter':
            self.pageSize = letter
        elif pageSize == 'A3':
            self.pageSize = A3

        self.width, self.height = self.pageSize

        self.pageSize = landscape(self.pageSize)

    def pageNumber(self, canvas, doc):
        number = canvas.getPageNumber()
        canvas.drawCentredString(100 * mm, 15 * mm, str(number))

    def report(self, report_df, title, banner_path):
        # prepare fancy report
        report_data = report_df.reset_index()
        # Rename several DataFrame columns
        report_data = report_data.rename(columns={
            'index': 'Structure',
            'constrain': 'Constrain',
            'constrain_value': 'Metric',
            'constrains_type': 'Constrain Type',
            'value_low': 'Lower Metric',
            'value_high': 'Upper Metric',
        })

        doc = SimpleDocTemplate(self.buffer,
                                rightMargin=9,
                                leftMargin=9,
                                topMargin=9,
                                bottomMargin=9,
                                pagesize=self.pageSize)

        # a collection of styles offer by the library
        styles = stylesheet()

        # list used for elements added into document
        data = []
        # add the banner
        data.append(Image(banner_path, width=doc.width * 0.99, height=doc.height * 0.2))
        data.append(Paragraph(title, styles['Participant Header']))
        # insert a blank space
        data.append(Spacer(1, 9))
        # first colun
        table_data = []
        # table header
        table_header = []
        for header in report_data.columns:
            table_header.append(Paragraph(header, styles['TableHeader']))

        table_data.append(table_header)

        i = 0
        for wh in report_data.values:
            # add a row to table
            ctr_tye = str(wh[3])
            if ctr_tye == 'upper':
                constrain_type = Paragraph(str(wh[3]), styles['upper'])
            else:
                constrain_type = Paragraph(str(wh[3]), styles['lower'])

            table_data.append(
                [Paragraph(str(wh[0]), styles['structure']),
                 Paragraph(str(wh[1]), styles['Text']),
                 Paragraph(str(wh[2]), styles['Text']),
                 constrain_type,
                 Paragraph("%0.2f" % wh[4], styles['number']),
                 Paragraph("%0.2f" % wh[5], styles['number']),
                 Paragraph("%0.2f" % wh[6], styles['number']),
                 Paragraph("%0.2f" % wh[7], styles['number']),
                 Paragraph("%0.2f" % wh[8], styles['number']),
                 Paragraph("{0} %".format(round(wh[9] * 100, 1)), styles['number'])])
            i += 1

        # adding last row
        total = report_data.values[:, 6].sum()
        score = report_data.values[:, 8].sum()
        performance = round(score / total * 100, 1)
        table_data.append(
            [None,
             None,
             None,
             None,
             None,
             Paragraph('Max Score:', styles['TextMax']),
             Paragraph("%0.2f" % total, styles['number']),
             Paragraph('Total Score', styles['Result number']),
             Paragraph("%0.2f" % score, styles['Result number']),
             Paragraph("{0} %".format(performance), styles['Result number'])])

        # create table
        wh_table = Table(data=table_data)
        wh_table.hAlign = 'LEFT'
        # wh_table.setStyle(TableStyle)
        wh_table.setStyle(TableStyle(
            [('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
             ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
             ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
             ('BACKGROUND', (0, 0), (-1, 0), colors.gray)]))
        data.append(wh_table)
        # data.append(Spacer(1, 48))
        # create document
        doc.build(data)


class FinalReportPDF(CompetitionReportPDF):
    def __init__(self, buffer, pageSize='A3'):
        CompetitionReportPDF.__init__(self, buffer, pageSize)

    def final_report(self, report_df, dose_stats_df, title, banner_path, dvh_path):
        # prepare fancy report
        report_data = report_df.reset_index()
        dose_stats_df = dose_stats_df.reset_index()
        # Rename several DataFrame columns
        report_data = report_data.rename(columns={
            'index': 'Structure',
            'constrain': 'Constrain',
            'constrain_value': 'Metric',
            'constrains_type': 'Constrain Type',
            'value_low': 'Lower Metric',
            'value_high': 'Upper Metric',
        })

        doc = SimpleDocTemplate(self.buffer,
                                rightMargin=9,
                                leftMargin=9,
                                topMargin=9,
                                bottomMargin=9,
                                pagesize=self.pageSize)

        # a collection of styles
        styles = stylesheet()

        # list used for elements added into document
        data = []
        # add the banner
        data.append(Image(banner_path, width=doc.width * 0.99, height=doc.height * 0.2))
        data.append(Paragraph(title, styles['Participant Header']))
        # insert a blank space
        data.append(Spacer(1, 9))
        # first colun
        table_data = []
        # table header
        table_header = []
        for header in report_data.columns:
            table_header.append(Paragraph(header, styles['TableHeader']))

        table_data.append(table_header)

        i = 0
        for wh in report_data.values:
            # add a row to table
            ctr_tye = str(wh[3])
            if ctr_tye == 'upper':
                constrain_type = Paragraph(str(wh[3]), styles['upper'])
            else:
                constrain_type = Paragraph(str(wh[3]), styles['lower'])

            table_data.append(
                [Paragraph(str(wh[0]), styles['structure']),
                 Paragraph(str(wh[1]), styles['Text']),
                 Paragraph(str(wh[2]), styles['Text']),
                 constrain_type,
                 Paragraph("%0.2f" % wh[4], styles['number']),
                 Paragraph("%0.2f" % wh[5], styles['number']),
                 Paragraph("%0.2f" % wh[6], styles['number']),
                 Paragraph("%0.2f" % wh[7], styles['number']),
                 Paragraph("%0.2f" % wh[8], styles['number']),
                 Paragraph("{0} %".format(round(wh[9] * 100, 1)), styles['number'])])
            i += 1

        # adding last row
        total = report_data.values[:, 6].sum()
        score = report_data.values[:, 8].sum()
        performance = round(score / total * 100, 1)
        table_data.append(
            [None,
             None,
             None,
             None,
             None,
             Paragraph('Max Score:', styles['TextMax']),
             Paragraph("%0.2f" % total, styles['number']),
             Paragraph('Total Score', styles['Result number']),
             Paragraph("%0.2f" % score, styles['Result number']),
             Paragraph("{0} %".format(performance), styles['Result number'])])

        # create table
        wh_table = Table(data=table_data)
        wh_table.hAlign = 'LEFT'
        # wh_table.setStyle(TableStyle)
        wh_table.setStyle(TableStyle(
            [('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
             ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
             ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
             ('BACKGROUND', (0, 0), (-1, 0), colors.gray)]))
        data.append(wh_table)
        # add page break
        data.append(PageBreak())

        # starting new page DVH stats
        # appendix = 'PyPlanScoring - Dose Volume Histogram - Calculation Results'
        #
        # data.append(Paragraph(appendix, styles['Participant Header']))
        # data.append(Spacer(1, 5))
        # DVH FIGURE
        data.append(Image(dvh_path, width=doc.width * .95, height=doc.height * .95))
        data.append(PageBreak())

        # table header
        dose_stats_df = dose_stats_df.rename(columns={
            'index': 'DVH Summary - Doses in cGy',
            'max': 'Maximum Dose',
            'mean': 'Average Dose',
            'min': 'Minimum Dose'
        })

        # Start DVH stats table
        dvh_table_data = []
        # table header
        dvh_table_header = []
        for header in dose_stats_df.columns:
            dvh_table_header.append(Paragraph(header, styles['TableHeader']))

        dvh_table_data.append(dvh_table_header)

        for wh in dose_stats_df.values:
            dvh_table_data.append(
                [Paragraph(str(wh[0]), styles['structure']),
                 Paragraph("%0.f" % wh[1], styles['number_dvh']),
                 Paragraph("%0.f" % wh[2], styles['number_dvh']),
                 Paragraph("%0.f" % wh[3], styles['number_dvh'])])

            i += 1

        # create table
        dvh_wh_table = Table(data=dvh_table_data)
        dvh_wh_table.hAlign = 'LEFT'
        # ALTERNAT ROW IN GRAY
        row_bg = [('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.lightgrey, colors.white]),
                  ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                  ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                  ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]

        table_style = TableStyle(row_bg)
        dvh_wh_table.setStyle(table_style)

        data.append(dvh_wh_table)

        # create document
        doc.build(data)


class PdfImage(Flowable):
    def __init__(self, img_data, width=200, height=200):
        self.img_width = width
        self.img_height = height
        self.img_data = img_data

    def wrap(self, width, height):
        return self.img_width, self.img_height

    def drawOn(self, canv, x, y, _sW=0):
        if _sW > 0 and hasattr(self, 'hAlign'):
            a = self.hAlign
            if a in ('CENTER', 'CENTRE', TA_CENTER):
                x += 0.5 * _sW
            elif a in ('RIGHT', TA_RIGHT):
                x += _sW
            elif a not in ('LEFT', TA_LEFT):
                raise ValueError("Bad hAlign value " + str(a))
        canv.saveState()
        canv.drawImage(self.img_data, x, y, self.img_width, self.img_height)
        canv.restoreState()


def make_report():
    fig = plt.figure(figsize=(4, 3))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 26])
    plt.ylabel('some numbers')
    imgdata = output
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)
    image = ImageReader(imgdata)

    doc = SimpleDocTemplate("hello.pdf")
    style = styles["Normal"]
    story = [Spacer(0, inch)]
    img = PdfImage(image, width=200, height=200)

    for i in range(10):
        bogustext = ("Paragraph number %s. " % i)
        p = Paragraph(bogustext, style)
        story.append(p)
        story.append(Spacer(1, 0.2 * inch))

    story.append(img)

    for i in range(10):
        bogustext = ("Paragraph number %s. " % i)
        p = Paragraph(bogustext, style)
        story.append(p)
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)


def get_random_colors(no_colors):
    # generate random hexa
    colors_list = []
    for i in range(no_colors):
        color = ''.join([choice('0123456789ABCDEF') for x in range(6)])
        colors_list.append(HexColor('#' + color))
    return colors_list
