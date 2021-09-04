# -*- coding: utf-8 -*-

"""
@author: shan
@software: PyCharm
@file: spilt_main.py
@time: 2021/9/4 2:26 下午
"""
# coding=utf-8

import os
import sys

kilobytes = 1024  # 1K byte
megabytes = kilobytes * 1000  # 1M byte
chunksize = int(200 * megabytes)  # default chunksize


def getPartSum(fromfile, chunksize):
    '''
    get the total number of part
    '''

    if os.path.getsize(fromfile) % chunksize != 0:
        return int(os.path.getsize(fromfile) / chunksize) + 1
    else:
        return int(os.path.getsize(fromfile) / chunksize)


def split(fromfile, todir, chunksize=chunksize):
    '''
    split files by the chunksize
    '''

    if not os.path.exists(todir):  # check whether todir exists or not
        os.mkdir(todir)  # make a folder
    else:
        for fname in os.listdir(todir):
            os.remove(os.path.join(todir, fname))
    partnum = 0  # the number of part
    partsum = getPartSum(fromfile, chunksize)  # the sum of parts
    inputfile = open(fromfile, 'rb')  # open the fromfile
    while True:
        chunk = inputfile.read(chunksize)
        if not chunk:  # check the chunk is empty
            break
        partnum += 1
        filename = os.path.join(todir, ('part%04d' % partnum))  # make file name
        fileobj = open(filename, 'wb')  # create partfile
        fileobj.write(bytes.fromhex('%04x' % partnum))  # write the serial number
        fileobj.write(bytes.fromhex('%04x' % partsum))  # write the sum of parts
        fileobj.write(chunk)  # write data into partfile
        fileobj.close()
    return partnum


if __name__ == '__main__':
    fromfile = input('File to be split?')
    todir = input('Directory to store part files?')
    chunksize = int(input('Chunksize to be split?'))
    absfrom, absto = map(os.path.abspath, [fromfile, todir])
    print('Splitting', absfrom, 'to', absto, 'by', chunksize)
    try:
        parts = split(fromfile, todir, chunksize)
    except:
        print('Error during split:')
        print(sys.exc_info()[0], sys.exc_info()[1])
    else:
        print('split finished:', parts, 'parts are in', absto)
