import os
from PIL import Image
import numpy as np

class PathFileGenerator:
    def __init__(self, rgb_base, flo_base, segment_base, label_base):
        self.rgb_base = rgb_base
        self.flo_base = flo_base
        self.segment_base = segment_base
        self.label_base = label_base

    def generate_BDEI_fashion(self, output_path, town_num, epi_num, img_num):
        """
        BDEI fashion means the path is formatted to the form like:
        base/downtown1/episode000/image_00001.png
        Note that in the path list, the downtown id start from 1 and end to town_num,
        while episode id start from 0 to epi_num-1, and
        image id start from 1 to img_num
        :param output_path: where to output path list file
        :param town_num: downtown id start from 1 and end to town_num
        :param epi_num: episode id start from 0 to epi_num-1
        :param img_num: file id start from 1 to img_num
        :return:
        """
        file = open(output_path, mode="w+")
        path_list = []
        for t in range(town_num):
            for e in range(epi_num):
                for i in range(1, img_num+1):
                    town_id = "downtown%d" % (t+1)
                    epi_id = "episode_%03d" % e
                    file_id = "image_%05d" % i
                    rgb_path = os.path.join(self.rgb_base, town_id, epi_id,
                                            "CameraRGB", file_id+".png")
                    segment_path = os.path.join(self.segment_base, town_id, epi_id,
                                                file_id+".png")
                    # formatted .flo file
                    flow_path = os.path.join(self.flo_base, town_id, epi_id,
                                             file_id +".flo")
                    label_path = os.path.join(self.label_base, town_id, epi_id,
                                              "Annotation", file_id+".png")
                    path_list.append(rgb_path+" "+segment_path+" "+flow_path+
                                     " "+label_path+"\n")
        file.writelines(path_list)

    def check_proportion(self, annotation_path):
        """
        Calculate the proportion of foreground
        :param annotation_path:
        :return:
        """
        img = Image.open(annotation_path)
        img = np.array(img.split()[0], dtype=np.float32)
        img = np.divide(img, np.array([255], dtype=np.float32))

        foreground = np.sum(img)
        all_img = img.shape[0]*img.shape[1]
        return foreground*1.0/all_img


    def generate_main_BDEI_fashion(self, output_path, town_num, epi_num,
                                   img_num, proportion):
        """
        This function will select those image whose foreground take a large proportion
        of background
        :param output_path: where to output path list file
        :param town_num: downtown id start from 1 and end to town_num
        :param epi_num: episode id start from 0 to epi_num-1
        :param img_num: file id start from 1 to img_num
        :param proportion: min proportion the foreground will take
        :return:
        """
        file = open(output_path, mode="w+")
        path_list = []
        for t in range(town_num):
            for e in range(epi_num):
                for i in range(1, img_num + 1):
                    town_id = "downtown%d" % (t + 1)
                    epi_id = "episode_%03d" % e
                    file_id = "image_%05d" % i
                    rgb_path = os.path.join(self.rgb_base, town_id, epi_id,
                                            "CameraRGB", file_id + ".png")
                    segment_path = os.path.join(self.segment_base, town_id, epi_id,
                                                file_id + ".png")
                    # formatted .flo file
                    flow_path = os.path.join(self.flo_base, town_id, epi_id,
                                             file_id + ".flo")
                    label_path = os.path.join(self.label_base, town_id, epi_id,
                                              "Annotation", file_id + ".png")
                    if self.check_proportion(label_path) > proportion:
                        path_list.append(rgb_path + " " + segment_path + " " + flow_path +
                                         " " + label_path + "\n")
        file.writelines(path_list)

    def generate_flo_path_BDEI_fashion(self, output_path, fflo_base, town_num, epi_num, img_num):
        """
        Generate .flo file path and .fflo file path respectively, which will be used
        to change .flo file to .fflo one
        :param output_path: where to output path list file
        :param town_num: downtown id start from 1 and end to town_num
        :param epi_num: episode id start from 0 to epi_num-1
        :param img_num: file id start from 1 to img_num
        :return:
        """
        file = open(output_path, mode="w+")
        path_list = []
        for t in range(town_num):
            for e in range(epi_num):
                for i in range(1, img_num + 1):
                    town_id = "downtown%d" % (t + 1)
                    epi_id = "episode_%03d" % e
                    file_id = "image_%05d" % i
                    fflo_path = os.path.join(fflo_base, town_id, epi_id,
                                             file_id + ".fflo")
                    flo_path = os.path.join(self.flo_base, town_id, epi_id,
                                             file_id + ".flo")
                    path_list.append(flo_path + " " + fflo_path + "\n")
        file.writelines(path_list)

base = "data_set-transNet"
rgb_base = os.path.join(base, "carla_rgb")
fflo_base = os.path.join(base, "carla_fflo")
segment_base = os.path.join(base, "carla_segment")
label_base = os.path.join(base, "carla_label")
flo_base = os.path.join(base, "carla_flo")
pfg = PathFileGenerator(rgb_base, flo_base, segment_base, label_base)
# pfg.generate_BDEI_fashion("data_path.txt", 1, 1, 899)
# pfg.generate_flo_path_BDEI_fashion("flo_path.txt", flo_base, 1, 1, 899)
pfg.generate_main_BDEI_fashion("data_path.txt", 1, 1, 899)



