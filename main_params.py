import os
import math
from pathlib import Path
import pathlib


class MainParams:
    def __init__(self):
        self.input_size = 102400 # 276480
        self.half_size = self.input_size // 2
        self.bin_size = 128
        self.hic_bin_size = 5000
        self.hic_size = 1176
        self.num_hic_bins = 50
        self.half_size_hic = (self.num_hic_bins // 2) * self.hic_bin_size
        self.num_bins = 400 # 1000
        self.half_num_regions = self.num_bins // 2
        self.mid_bin = self.num_bins // 2
        self.BATCH_SIZE = 1
        self.NUM_GPU = 2
        self.GLOBAL_BATCH_SIZE = self.NUM_GPU * self.BATCH_SIZE
        self.predict_batch_size = self.NUM_GPU * 2 * self.BATCH_SIZE
        self.w_step = 200
        self.STEPS_PER_EPOCH = 800
        self.num_epochs = 1000
        self.num_features = 4
        self.species = ["hg38", "mm10", "macFas5", "calJac4", "rheMac8", "canFam3", "oviAri4", "rn6"]
        # self.species = ["hg38"]
        self.script_folder = pathlib.Path(__file__).parent.resolve()
        folders = open(str(self.script_folder) + "/../data_dirs").read().strip().split("\n")
        self.data_folder = folders[0]
        os.chdir(self.data_folder)
        self.parsed_tracks_folder = folders[1]
        self.hic_folder = folders[2]
        self.model_folder = folders[3]
        self.temp_folder = folders[4]
        self.pickle_folder = folders[5]
        self.model_name = "our_model_276480" # "our_model_small.h5"
        self.model_path = self.model_folder + self.model_name
        self.figures_folder = "figures_1"
        self.tracks_folder = "/media/user/PASSPORT1/variants_100/tracks/"
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "plots").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)