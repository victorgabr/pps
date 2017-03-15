import json
import mimetypes
import os

import dicom
from flask import Flask, make_response, request, render_template, abort
from werkzeug.datastructures import Headers

from web_app.dosevue.report_creator import PlanReportCreator

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/v1/get_structure_names/", methods=["POST"])
def process_structures():
    if request.method == "POST":
        file_stream = request.files["files[]"]
        file_stream.save(file_stream.filename)
        rs = dicom.read_file(file_stream.filename)

        try:
            struct_list = [{"name": structure.ROIName, "value": index} for index, structure in
                           enumerate(rs.StructureSetROISequence) if "ROIName" in structure]

            response = {
                "structure_list": struct_list,
                "rs_filename": file_stream.filename
            }
        except AttributeError:
            abort(500)

        return make_response(json.dumps(response))
    else:
        return make_response("Expected POST request.", 422)


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


@app.route("/api/v1/upload_ct/", methods=["POST"])
def upload_ct():
    if request.method == "POST":
        file_stream = request.files["files[]"]
        file_stream.save(file_stream.filename)

        response = {
            "ct_filename": file_stream.filename
        }

        return make_response(json.dumps(response))
    else:
        return make_response("Expected POST request.", 422)


@app.route("/api/v1/create_report/", methods=["POST"])
def report_creator():
    structure_file_name = request.form["rs_filename"]
    dose_file_names = json.loads(request.form["rd_filename"])
    ct_file_name = request.form["ct_filename"]
    # roi_numbers = [int(request.form["roi"])]
    # report = ReportCreator()
    # Plan report
    report = PlanReportCreator()

    status = report.loadStruct(structure_file_name)
    if status != 0:
        return make_response(status, 422)

    status = report.loadDose(dose_file_names[0])
    if status != 0:
        return make_response(status, 422)

    report.load_plan(ct_file_name)

    # report.set_ctfile(ct_file_name)
    # report.loadModel(roi_numbers)
    f = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/Scoring Criteria.txt'
    report_string = report.makeReport(f)
    response = make_response(report_string)
    # response.headers["Content-Disposition"] = "inline; filename=mandible.pdf"
    # response.mimetype = "application/pdf"




    os.remove(structure_file_name)
    for dose_filename in dose_file_names:
        os.remove(dose_filename)

    os.remove(ct_file_name)

    ################################
    # Code for setting correct
    # headers for jquery.fileDownload
    #################################
    filename = "Report.xlsx"
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
        response.update({
            'Content-Encoding': mimetype_tuple[1]
        })

    response.headers = response_headers

    # as per jquery.fileDownload.js requirements
    response.set_cookie('fileDownload', 'true', path='/')

    ################################
    # Return the response
    #################################
    return response


if __name__ == "__main__":
    app.debug = True
    app.run()
