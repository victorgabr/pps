import json
import mimetypes
import os

from flask import Flask, make_response, request, render_template
from werkzeug.datastructures import Headers

from web_app.dosevue.report_creator import PlanReportCreator

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/api/v1/get_structure_names/", methods=["POST"])
# def process_structures():
#     if request.method == "POST":
#         file_stream = request.files["files[]"]
#         file_stream.save(file_stream.filename)
#         rs = dicom.read_file(file_stream.filename)
#
#         try:
#             struct_list = [{"name": structure.ROIName, "value": index} for index, structure in
#                            enumerate(rs.StructureSetROISequence) if "ROIName" in structure]
#
#             response = {
#                 "structure_list": struct_list,
#                 "rs_filename": file_stream.filename
#             }
#         except AttributeError:
#             abort(500)
#
#         return make_response(json.dumps(response))
#     else:
#         return make_response("Expected POST request.", 422)


@app.route("/api/v1/upload_dose/", methods=["POST"])
def upload_dose():
    if request.method == "POST":
        file_stream = request.files["files[]"]
        file_stream.save(file_stream.filename)

        response = {
            "rd_filename": file_stream.filename
        }

        return make_response(json.dumps(response))
    else:
        return make_response("Expected POST request.", 422)


# #
# @app.route("/api/v1/upload_ct/", methods=["POST"])
# def upload_ct():
#     if request.method == "POST":
#         file_stream = request.files["files[]"]
#         file_stream.save(file_stream.filename)
#
#         response = {
#             "ct_filename": file_stream.filename
#         }
#
#         return make_response(json.dumps(response))
#     else:
#         return make_response("Expected POST request.", 422)


@app.route("/api/v1/create_report/", methods=["POST"])
def report_creator():
    # get report files
    work_dir = os.getcwd()
    banner_path = os.path.join(work_dir, '2017 Plan Comp Banner.jpg')
    criteria_txt = os.path.join(work_dir, 'Scoring Criteria.txt')
    structure_file_name = os.path.join(work_dir, 'RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm')
    plan_file_name = os.path.join(work_dir, 'RP.1.2.246.352.71.5.584747638204.955801.20170210152428.dcm')

    dose_file_name = request.form["rd_filename"]

    # Initialize Plan Report Class
    report = PlanReportCreator()

    status = report.loadStruct(structure_file_name)
    # if status != 0:
    #     return make_response(status, 422)

    status = report.loadDose(dose_file_name)
    if status != 0:
        return make_response(status, 422)

    # load plan file
    report.load_plan(plan_file_name)

    # calculate DVH and return excel report

    report_data = report.makeReport(criteria_txt, banner_path=banner_path)
    response = make_response(report_data)

    # clean up uploaded files
    # os.remove(dose_file_name)

    ################################
    # Code for setting correct
    # headers for jquery.fileDownload
    #################################

    p, file_name = os.path.split(dose_file_name)

    filename = file_name + "_PlanReport.xlsx"
    mimetype_tuple = mimetypes.guess_type(filename)

    # HTTP headers for forcing file download
    response_headers = Headers({
        'Pragma': "public",  # required,
        'Expires': '0',
        'Cache-Control': 'must-revalidate, post-check=0, pre-check=0',
        'Cache-Control': 'private',  # required for certain browsers,
        'Content-Type': mimetype_tuple[0],
        'Content-Disposition': 'attachment; filename=\"%s\";' % filename,
        'Content-Transfer-Encoding': 'binary',
        'Content-Length': len(response.data)
    })

    if not mimetype_tuple[1] is None:
        response.update({'Content-Encoding': mimetype_tuple[1]})

    response.headers = response_headers
    # as per jquery.fileDownload.js requirements
    response.set_cookie('fileDownload', 'true', path='/')

    return response


if __name__ == "__main__":
    app.debug = True
    app.run()
