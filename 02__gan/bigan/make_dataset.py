# corexec -pjxcontents jcl_kw.txt -python make_dataset.py

from proteus import simulation as psim
from proteus import fops
from proteus import layer as proteus_layer
from proteus import layout as proteus_layout
#from proteus import info as proteusinfo
from proteus.fops import kernels
import pandas as pd
import numpy as np
import os, sys
from collections import OrderedDict
import argparse


class Simulator(object):
    def __init__(self, model_file, test_pattern, layer_map, field_matrix_size):
        self.test_pattern = test_pattern
        self.layout_holder = proteus_layout.LayoutHolder(test_pattern)
        self.layer_map = layer_map
        self.layer_dt = [tuple(map(int, ld.split(":"))) for ld in layer_map.values()]
        self.shape = field_matrix_size
        # construct simulator
        self._modeldef = psim.makeSimulator({"image"}, model_file)
        self.pitch = self._modeldef.sampling_pitch / 2.0
        self.ambit = self._modeldef.ambit
        self.msf = self._modeldef.msf
        self.threshold = self._modeldef.model_parameters['thresh_const']

    def _get_clip(self, bbox, offset=None):
        w, s, e, n = bbox
        clip = proteus_layer.readOasGds(self.layout_holder, 
            layer_dt=self.layer_dt,
            offset=offset, 
            clip_box=bbox,
            shape_type='auto',
            new_api=True)
        return clip

    def __getLayers(self, key, clip):
        ld = self.layer_map[key]
        ld_vals = tuple(map(int, ld.split(":")))
        return clip[ld_vals]

    def sim(self, clip, origin):
        sim_graphics = {key: self.__getLayers(key, clip) for key in self.layer_map.keys()}
        grid = fops.Field(origin=origin, shape=self.shape, pitch=self.pitch)
        self.simulator = self._modeldef(graphics=sim_graphics, grid=grid)
    
    def get_field_values(self):
        # get field values on fops grid
        field_values = self.simulator.image.signal.field.value
        return field_values

    def resampler(self, x_list, y_list):
        # get signal at arbitrary coord
        fval = self.simulator.image.signal.at(x_list, y_list)
        return fval



def main():

    # model_file = "/remote/ltg_proj02_us01/user/sanghoon/AEI_REVERSE/gaus_rasterization_amdl/vt0_gausRast06.amdl"
    # asd_file = "/remote/ltg_proj02_us01/user/sanghoon/etch_modeling/datasets/rich3/EPall_80478.asd"
    # layout_file = "/remote/ltg_proj02_us01/user/sanghoon/etch_modeling/datasets/rich3/n2_case12AEI_mimic.oas"

    layer_map_adi = OrderedDict()
    layer_map_adi['main_in']=args.layer_map_adi
    layer_map_aei = OrderedDict()
    layer_map_aei['main_in']=args.layer_map_aei

    field_matrix_size = (512, 512)
    
    sim_adi = Simulator(args.model_file, args.layout_file, layer_map_adi, field_matrix_size)
    sim_aei = Simulator(args.model_file, args.layout_file, layer_map_aei, field_matrix_size)

    A = pd.read_csv(args.asd_file, sep="\s+", comment="'")
    pad = 0
    clip_size = field_matrix_size[0] * sim_adi.pitch

    image_size = 309
    yy = (image_size//2)*4
    xx = -yy
    tp = np.linspace(xx, yy, image_size)
    X, Y = np.meshgrid(tp, tp)

    for j, (baseX, baseY) in enumerate(zip(A.base_x.values, A.base_y.values)):
        ll = [baseX-clip_size//2, baseY-clip_size//2]
        ur = [baseX+clip_size//2, baseY+clip_size//2]
        w, s, e, n = (ll[0]-pad, ll[1]-pad, ur[0]+pad, ur[1]+pad)
        bbox = (w, s, e, n)

        # adi
        clip_adi = sim_adi._get_clip(bbox, offset=(-baseX, -baseY))
        sim_adi.sim(clip_adi, origin=(-clip_size//2, -clip_size//2))
        adi = sim_adi.simulator.image.signal.at(X.reshape(-1), Y.reshape(-1))
        adi = adi.reshape([image_size,image_size])

        # aei
        clip_aei = sim_aei._get_clip(bbox, offset=(-baseX, -baseY))
        sim_aei.sim(clip_aei, origin=(-clip_size//2, -clip_size//2))
        aei = sim_aei.simulator.image.signal.at(X.reshape(-1), Y.reshape(-1))
        aei = aei.reshape([image_size,image_size])
        
        img = np.concatenate((adi, aei), axis=-1) 
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        np.save("{}/{}_{}".format(args.save_path, baseX, baseY), img)
        print "complete gauge {}, image center at ({}, {}), img.shape: {}".format(j+1, baseX, baseY, img.shape)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="etch modeling dataset creation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_file", type=str, required=True, help="rasterization model")
    parser.add_argument("--asd_file", type=str, required=True)
    parser.add_argument("--layout_file", type=str, required=True)
    parser.add_argument("--layer_map_adi", type=str, required=True)
    parser.add_argument("--layer_map_aei", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    main()
