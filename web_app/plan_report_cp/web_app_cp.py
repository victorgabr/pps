import os

import cherrypy

TEMP_DIR = os.path.abspath(os.path.join(os.getcwd(), 'templates'))


class Root(object):
    @cherrypy.expose
    def index(self):
        return open(os.path.join(TEMP_DIR, 'index.html'))

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def upload_dose(self):
        print(cherrypy.request)

    @cherrypy.expose
    def create_report(self):
        pass


if __name__ == '__main__':
    cherrypy.config.update({'server.socket_port': 8090})

    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': 'static'
        }
    }

    cherrypy.quickstart(Root(), '/', conf)
