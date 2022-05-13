from constants import *

def listostr2arr(sls):
    listofstring = sls[1:-1].split(",")
    arrofstring = []
    for i in range(0, len(listofstring), 3):
        arrofstring.append(listofstring[i] + "," + listofstring[i + 1] + "," + listofstring[i + 2])
    steps = []
    for i in arrofstring:
        i = "np." + (i[1:] if i[0] == "[" else i).strip()
        i = i if i[-1] != "]" else i[:-1]

        steps.append(eval(i))
    return steps

def str2tuple(s):
    s = s[1:-1]
    s = s.split(" ")
    s = [i for i in s if i != ""]
    return tuple(map(float, s))
    # s = "(" + ",".join(s) + ")"
    # return tuple(eval(s))


def strarr2tuple(s):
    s = s[1:-1]
    if "," in s:
        s = s.split(",")
    else:
        s = s.split(" ")
    s = [i for i in s if i != ""]

    return tuple(map(float, s))

def unzip_data():
    import time
    from constants import root
    import zipfile
    import os
    datadir = root.joinpath("Data")
    # unzip everythin in datadir

    for file in datadir.glob('*'):
        if file.suffix == '.zip':
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(datadir)

    for file in datadir.glob('*'):
        if file.suffix == '.zip':
            file.unlink()

def zip_data():
    import shutil
    import zipfile
    from constants import root
    import os
    datadir = root.joinpath('Data')

    # make directory of zips of files in data
    if not datadir.exists():
        datadir.mkdir()
    for file in datadir.glob('*'):
        if file.suffix == '.zip':
            continue
        with zipfile.ZipFile(file.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zip:
            zip.write(file, arcname=file.name)
        file.unlink()
    print("Zip done")


