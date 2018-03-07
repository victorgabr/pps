# -*- mode: python -*-

block_cipher = None
import os

folder = os.getcwd()

docfiles = os.path.join(folder, 'licence.txt')
banner = os.path.join(folder, '2017 Plan Comp Banner.jpg')
criteria = os.path.join(folder, 'Scoring Criteria.txt')
rs_file = os.path.join(folder, 'RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm')

added_files = [
    (docfiles, '.'),
    (banner, '.'),
    (criteria, '.'),
    (rs_file, '.'),
]

a = Analysis(['app.py'],
             pathex=[r'C:\Users\Victor\Dropbox\Plan_Competition_Project',
                     r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring'],
             binaries=[],
             datas=added_files,
             hiddenimports=[],
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
