import mimetypes

import xlwt
from flask import Response
from werkzeug.datastructures import Headers


# ... code for setting up Flask

@app.route('/export/')
def export_view():
    #########################
    # Code for creating Flask
    # response
    #########################
    response = Response()
    response.status_code = 200

    ##################################
    # Code for creating Excel data and
    # inserting into Flask response
    ##################################
    workbook = xlwt.Workbook()

    # .... code here for adding worksheets and cells

    output = StringIO.StringIO()
    workbook.save(output)
    response.data = output.getvalue()

    ################################
    # Code for setting correct
    # headers for jquery.fileDownload
    #################################
    filename = export.xls
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


import SimpleHTTPServer
import SocketServer
import StringIO

import xlsxwriter


class Handler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Create an in-memory output file for the new workbook.
        output = StringIO.StringIO()

        # Even though the final file will be in memory the module uses temp
        # files during assembly for efficiency. To avoid this on servers that
        # don't allow temp files, for example the Google APP Engine, set the
        # 'in_memory' constructor option to True:
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        worksheet = workbook.add_worksheet()

        # Write some test data.
        worksheet.write(0, 0, 'Hello, world!')

        # Close the workbook before streaming the data.
        workbook.close()

        # Rewind the buffer.
        output.seek(0)

        # Construct a server response.
        self.send_response(200)
        self.send_header('Content-Disposition', 'attachment; filename=test.xlsx')
        self.send_header('Content-type',
                         'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        self.end_headers()
        self.wfile.write(output.read())
        return


print('Server listening on port 8000...')
httpd = SocketServer.TCPServer(('', 8000), Handler)
httpd.serve_forever()
