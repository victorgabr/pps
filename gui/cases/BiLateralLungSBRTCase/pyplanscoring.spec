# -*- mode: python -*-

block_cipher = None
import os

folder = os.getcwd()

docfiles = os.path.join(folder, 'licence.txt')
criteria = os.path.join(folder, 'Scoring_criteria.xlsx')
rs_file = os.path.join(folder, 'RS_LUNG_SBRT.dcm')

added_files = [
    (docfiles, '.'),
    (banner, '.'),
    (criteria, '.'),
    (rs_file, '.'),
]

a = Analysis(['app.py'],
             pathex=[r'D:\Plan_Competition_Project\gui',
                       r'D:\Plan_Competition_Project',
                        r'D:\Plan_Competition_Project\pyplanscoring'],
             binaries=[],
             datas=added_files,
             hiddenimports=['pandas._libs.tslibs.timedeltas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             )
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='PyPlanScoring',
          debug=False,
          strip=False,
          upx=True,
          console=True)
