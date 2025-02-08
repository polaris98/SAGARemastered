import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel

import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

import os
import utils.system_utils as system_utils
# The 2D SAM masks are projected into 3D by using the depth map and the camera intrinsics. 
# This yields a point cloud for each mask.

# The 3D point cloud for each mask is analyzed to compute its spatial spread (using standard deviation and 
# Euclidean norm), resulting in a scale sm.

# This scale information is then used in the SAGA pipeline to adjust the Gaussian affinity features via a scale gate. 
# The scale gating helps resolve multi-granularity ambiguity, meaning that the same 3D Gaussian can belong to different 
# segmentation parts at different scales.

FEATURE_DIM = 32

DATA_ROOT = './data/nerf_llff_data_for_3dgs/'
# MODEL_PATH = './output/figurines_lerf_poses/'
# MODEL_PATH = './output/figurines/'

ALLOW_PRINCIPLE_POINT_SHIFT = False


def get_combined_args(parser : ArgumentParser):
    # cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()
    
    target_cfg_file = "cfg_args"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    # for k in args_cfgfile.__dict__.keys():
        # print(k, args_cfgfile.__dict__[k], "?")

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    # for k in merged_dict.keys():
        # print(k, merged_dict[k])
    return Namespace(**merged_dict)

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)

    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False

    # ALLOW_PRINCIPLE_POINT_SHIFT = 'lerf' in args.model_path
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    assert os.path.exists(os.path.join(dataset.source_path, 'images')) and "Please specify a valid image root."
    assert os.path.join(dataset.source_path, 'sam_masks') and "Please run extract_segment_everything_masks first."

    from tqdm import tqdm
    images_masks = {}
    for i, image_path in tqdm(enumerate(sorted(os.listdir(os.path.join(dataset.source_path, 'images'))))):
        # print(image_path)
        image = cv2.imread(os.path.join(os.path.join(dataset.source_path, 'images'), image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = torch.load(os.path.join(os.path.join(dataset.source_path, 'sam_masks'), image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')))
        # N_mask, C

        images_masks[image_path.split('.')[0]] = masks.cpu().float()
    
    feature_gaussians = None
    scene_gaussians = GaussianModel(dataset.sh_degree)
   
    iter_list = system_utils.searchForIteration(os.path.join(dataset.model_path, "point_cloud"), target="scene", iterationMax=args.iteration)
   
    for iter in iter_list:
        scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=iter, feature_load_iteration=iter, shuffle=False, mode='eval', target='scene')
        
        if iter == iter_list[0]:
            print("inside")
            cameras = scene.getTrainCameras()

        background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device = 'cuda')

        for it, view in tqdm(enumerate(cameras)):
            
            #returns a dictionary including a depth map for the current view
            #tells how fare each pixel is from the camera
            rendered_pkg = gaussian_renderer.render_with_depth(view, scene_gaussians, pipeline.extract(args), background)

            depth = rendered_pkg['depth']

            # plt.imshow(depth.detach().cpu().squeeze().numpy())

            #The code SAM masks (stored earlier) that correspond to the current view
            corresponding_masks = images_masks[view.image_name]

            # generate_grid_index(depth.squeeze())[50, 1]
            
            depth = depth.cpu().squeeze()

            #creates a 2D grid of pixel coordinates matching the depth map’s dimensions.
            grid_index = generate_grid_index(depth)

            points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
            points_in_3D[:,:,-1] = depth

            # caluculate cx cy fx fy with FoVx FoVy
            # The x and y coordinates are computed using the pixel coordinates (from the grid), 
            # the camera’s principal point (cx, cy), and the focal lengths (fx, fy).

            # convert 2D pixel locations into 3D coordinates in the camera’s coordinate system,
            # enabling to form a 3D point cloud of the scene from the depth map and image grid.
            cx = depth.shape[1] / 2
            cy = depth.shape[0] / 2
            fx = cx / np.tan(cameras[0].FoVx / 2)
            fy = cy / np.tan(cameras[0].FoVy / 2)


            points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
            points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy

            #The SAM mask (which may have lower resolution) is upsampled (using bilinear interpolation) to match the depth map’s resolution.
            upsampled_mask = torch.nn.functional.interpolate(corresponding_masks.unsqueeze(1), mode = 'bilinear', size = (depth.shape[0], depth.shape[1]), align_corners = False)

            #remove noisy boundaries
            eroded_masks = torch.conv2d(
                upsampled_mask.float(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3),
                padding=1,
            )
            eroded_masks = (eroded_masks >= 5).squeeze()  # (num_masks, H, W)

            scale = torch.zeros(len(corresponding_masks))
            for mask_id in range(len(corresponding_masks)):

                # select the 3D points from points_in_3D that fall inside the eroded mask
                # This gives a point cloud P for that object
                point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]

                #calculates the standard deviation along each 3D axis of the points in P
                scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

            OUTPUT_DIR = os.path.join(args.image_root, 'mask_scales')
            os.makedirs(OUTPUT_DIR + "/iteration_"+str(iter), exist_ok=True)
            torch.save(scale, os.path.join(OUTPUT_DIR + "/iteration_"+str(iter)+"/", view.image_name + '.pt'))