# -*- coding: utf-8 -*-
import os.path
import cherrypy
from jinja2 import Environment, FileSystemLoader
from web_app.jinja2_templating.jinja2plugin import Jinja2TemplatePlugin, J


class Root(object):
    @cherrypy.expose
    def index(self):
        return {'msg': 'Hello world!'}


if __name__ == '__main__':
    cherrypy.config.update({'server.socket_port': 8090})
    # Register the Jinja2 plugin

    env = Environment(loader=FileSystemLoader('.'))
    Jinja2TemplatePlugin(cherrypy.engine, env=env).subscribe()

    # Register the Jinja2 tool
    from jinja2tool import Jinja2Tool

    cherrypy.tools.template = Jinja2Tool()

    # We must disable the encode tool because it
    # transforms our dictionary into a list which
    # won't be consumed by the jinja2 tool
    cherrypy.quickstart(Root(), '', {'/': {'tools.template.on': True,
                                           'tools.template.template': 'index.html',
                                           'tools.encode.on': False}})
