import struct
import os

def parse_flo_file(input_path, output_path):
    """
    This function read data from .flo file, whose data are formed
    as h(0,0) v(0,0) h(0,1) v(0,1)..., while h and v is the horizontal
    and vertical speed of pixel respectively
    And output the data formed like a 2-channel image file, where one
    for horizontal speed and the other for vertical one
    :param input_path:
    :param output_path:
    :return:
    """
    file = open(input_path, "rb")
    (_, cols_num, rows_num) = struct.unpack("4s2i", file.read(12))

    # flow information is sort as h(0,0) v(0,0) h(0,1) v(0,1)...
    horizontal_flow = []
    vertical_flow = []
    for r in range(rows_num):
        row_horizontal = []
        row_vertical = []
        for c in range(cols_num):
            h, v = struct.unpack("ff", file.read(8))
            row_horizontal.append(h)
            row_vertical.append(v)
        horizontal_flow.append(row_horizontal)
        vertical_flow.append(row_vertical)

    file.close()
    #save as fflo
    output_dir = output_path.split("/")[0:-1]
    output_dir = os.path.join(*output_dir)
    if not os.path.exists(output_dir):
        print output_dir
        os.makedirs(output_dir)
    file = open(output_path, "wb+")
    size_data = struct.pack("ii", rows_num, cols_num)
    file.write(size_data)
    for row in horizontal_flow:
        h_data = struct.pack("%df" % cols_num, *row)
        file.write(h_data)
    for row in vertical_flow:
        v_data = struct.pack("%df" % cols_num, *row)
        file.write(v_data)
    file.close()

def check(flo_path, fflo_path):
    """
    check wether the parse function coded correctly
    :param flo_path:
    :param fflo_path:
    :return:
    """
    flo_file = open(flo_path, "rb")
    (_, cols_num, rows_num) = struct.unpack("4s2i", flo_file.read(12))
    fflo_file = open(fflo_path, "rb")
    (_rows_num, _cols_num) = struct.unpack("2i", fflo_file.read(8))
    if cols_num != _cols_num or rows_num != _rows_num:
        print "cols OR rows num is wrong"
        return

    # read .flo file out
    horizontal_flow = []
    vertical_flow = []
    for r in range(rows_num):
        row_horizontal = []
        row_vertical = []
        for c in range(cols_num):
            h, v = struct.unpack("ff", flo_file.read(8))
            row_horizontal.append(h)
            row_vertical.append(v)
        horizontal_flow.append(row_horizontal)
        vertical_flow.append(row_vertical)
    # do compare
    for r in range(rows_num):
        for c in range(cols_num):
            # note that the comma here change h from tuple to a float
            h, = struct.unpack("f", fflo_file.read(4))
            if h != horizontal_flow[r][c]:
                print "horizontal data wrong"
                return
    for r in range(rows_num):
        for c in range(cols_num):
            v, = struct.unpack("f", fflo_file.read(4))
            if v != vertical_flow[r][c]:
                print "vertical data wrong"
                return
    print "right"

def format_flo_to_fflo(change_file):
    file = open(change_file)
    lines = file.readlines()
    for line in lines:
        flo_path = line.split()[0]
        fflo_path = line.split()[1]
        parse_flo_file(flo_path, fflo_path)



format_flo_to_fflo("flo_path.txt")


