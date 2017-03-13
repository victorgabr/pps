# -*- mode: python -*-

import os

block_cipher = None


def Datafiles(*filenames, **kw):
    def datafile(path, strip_path=True):
        parts = path.split('/')
        path = name = os.path.join(*parts)
        if strip_path:
            name = os.path.basename(path)
        return name, path, 'DATA'

    strip_path = kw.get('strip_path', True)
    return TOC(
        datafile(filename, strip_path=strip_path)
        for filename in filenames
        if os.path.isfile(filename))


workpath = r'C:\PYTHON_BUILD\build'

DISTPATH = r'C:\PYTHON_BUILD\dist'

folder = os.getcwd()

docfiles = Datafiles(os.path.join(folder, 'licence.txt'))
banner = Datafiles(os.path.join(folder, '2017 Plan Comp Banner.jpg'))
criteria = Datafiles(os.path.join(folder, 'Scoring Criteria.txt'))
rs_file = Datafiles(os.path.join(folder, 'RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm'))

a = Analysis(['app.py'],
             pathex=[r'C:\Users\Victor\Dropbox\Plan_Competition_Project',
                     r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='PyPlanScoring',
          debug=False,
          strip=False,
          upx=True,
          console=True)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               docfiles,
               banner,
               criteria,
               rs_file,
               strip=False,
               upx=True,
               name='PyPlanScoring')





# pyi-makespec --distpath r'C:\PYTHON_BUILD\dist' --workpath r'C:\PYTHON_BUILD\build' app.py
