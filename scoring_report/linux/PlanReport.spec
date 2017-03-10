# -*- mode: python -*-

block_cipher = None
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


docfiles = Datafiles(r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/licence.txt')
mkl = Datafiles(r'C:\Users\Victor\Dropbox\DFR\win64\mkl_avx.dll')
# mkl1 = Datafiles(r'/home/victor/Dropbox/DFR/win64/mkl_def.dll')
# mkl2 = Datafiles(r'C:\Users\Victor\Dropbox\DFR\win64\mkl_mc.dll')
banner = Datafiles(r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/2017 Plan Comp Banner.jpg')
criteria = Datafiles(r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/Scoring Criteria.txt')
readme = Datafiles(r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/README.txt')

a = Analysis([r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/PlanReport.py'],
             pathex=[r'/home/victor/Dropbox/Plan_Competition_Project',
                     r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring'],
             binaries=None,
             datas=None,
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
          name='PlanReport',
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
               # mkl1,
               criteria,
               readme,
               strip=False,
               upx=True,
               name='PlanReport')
