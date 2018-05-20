# -*- mode: python -*-

block_cipher = None

a = Analysis(['viewer.py'],
             pathex=['D:\\Dropbox\\Plan_Competition_Project\\web_app\\plan_report'
                     r'D:\Dropbox\Plan_Competition_Project\pyplanscoring',
                     r'D:\Dropbox\Plan_Competition_Project'],
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
          name='viewer',
          debug=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='viewer')
