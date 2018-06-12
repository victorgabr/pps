# this module is for experimentation using notebook or console.

# from pyplanscoring import PyDicomParser

import pydicom

if __name__ == '__main__':
    filename = r"C:\Users\vgalves\Downloads\monacoplangetalldetails\2018CompLung_lmh.dcm"
    obj = pydicom.read_file(filename, force=True)
    pass
