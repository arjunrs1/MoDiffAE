#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
#from PyMoCapViewer import MoCapViewer
from visualize.PyMoCapViewer.PyMoCapViewer.mocap_viewer import MoCapViewer
from utils.karate import data_info
import preprocessing.karate.data_prep as data_prep
import numpy as np
import math
import os
import shutil
import time 

import vtk
import open3d as o3d


def get_labels(): 
    labels = []
    label_to_index = {}
    count = 0
    for l in data_info.joint_to_index.keys():
        if 'BACK' in l:
            l = l.replace('BACK', 'RBAK')
        labels.append(l + ' (x)')
        label_to_index[l + ' (x)'] = count
        count += 1
        labels.append(l + ' (y)')
        label_to_index[l + ' (y)'] = count
        count += 1
        labels.append(l + ' (z)')
        label_to_index[l + ' (z)'] = count
        count += 1
    return labels, label_to_index


def determine_nr_of_pcd_frames(nr_frames):
    if nr_frames % 10 == 0:
        # Every keyframe gets the same number of point clouds
        pcd_nr_frames_middle = int((nr_frames / 10.0) - 1) #10
        pcd_nr_frames_start = int(pcd_nr_frames_middle)
        pcd_nr_frames_end = int(pcd_nr_frames_middle)
    else:
        # Spread as much as possible among the middle 8 and 
        # then redistribute in case of the existence of more even splits,
        # while putting more importance on the middle.
        pcd_nr_frames_middle = math.floor(nr_frames / 8.0) - 1
        rest = nr_frames % 8
        pcd_nr_frames_start = math.floor(rest / 2) - 1
        pcd_nr_frames_end = math.ceil(rest / 2) - 1

        if pcd_nr_frames_middle < 0:
            pcd_nr_frames_middle = 0
            pcd_nr_frames_start = 0
            pcd_nr_frames_end = 0

        if (pcd_nr_frames_start < 0 or pcd_nr_frames_end < 0):
            if pcd_nr_frames_middle > 1:
                pcd_nr_frames_middle -= 1
                pcd_nr_frames_start += 4
                pcd_nr_frames_end += 4
            else:     
                pcd_nr_frames_middle = 0
                pcd_nr_frames_start = 0
                pcd_nr_frames_end = 0

        while pcd_nr_frames_middle >= pcd_nr_frames_start + 4:
            pcd_nr_frames_middle -= 1
            pcd_nr_frames_start += 4
            pcd_nr_frames_end += 4

    return pcd_nr_frames_start, pcd_nr_frames_middle, pcd_nr_frames_end


def fade_pointclouds(colors):
    nr_of_pcd_frames = colors.shape[0]
    if nr_of_pcd_frames != 0:

        if nr_of_pcd_frames == 1:
            colors[0, :] = 0.2
        else:
            fade_min = 0.3 # 0.2
            fade_max = 0.95
            step_size = (fade_max - fade_min) / (nr_of_pcd_frames - 1)
            for i in range(nr_of_pcd_frames):
                colors[i, :] = fade_min + (nr_of_pcd_frames - i - 1) * step_size
                
    return colors


def prepare_for_static_mode(df):
    df_copy = df.copy()
    
    offsets_x = []
    offsets_y = []

    pcd_list = []

    nr_frames = df.shape[0]

    pcd_nr_frames_start, pcd_nr_frames_middle, pcd_nr_frames_end = \
        determine_nr_of_pcd_frames(nr_frames)

    first_keyframe_idx = pcd_nr_frames_start
    last_keyframe_idx = df.shape[0] - 1
    idx_counter = first_keyframe_idx
    keyframes = [first_keyframe_idx]
    while len(keyframes) < 9:
        idx_counter += pcd_nr_frames_middle + 1
        keyframes.append(idx_counter)
    keyframes.append(last_keyframe_idx)
    
    print(pcd_nr_frames_start, pcd_nr_frames_middle, pcd_nr_frames_end, nr_frames)
    print(keyframes)
    
    for frame in range(df.shape[0]):

        if frame == df.shape[0] - 1:
            # Last keyframe
            pcd_nr_frames = pcd_nr_frames_end
        elif frame <= pcd_nr_frames_start:
            # Before the first keyframe or the first one
            pcd_nr_frames = pcd_nr_frames_start
        else:
            # In the middle
            pcd_nr_frames = pcd_nr_frames_middle

        if frame < pcd_nr_frames:
            start = 0
        else: 
            start = frame - pcd_nr_frames
        pcd_frame = df_copy.loc[start:frame - 1, :].copy()

        offset_x = df.loc[frame, 'STRN (x)'].copy()
        offsets_x.append(offset_x)
        offset_y = df.loc[frame, 'STRN (y)'].copy()
        offsets_y.append(offset_y)
        labels, _ = get_labels()
        for j in [l for l in labels if '(x)' in l]:
            df.loc[frame, j] = df.loc[frame, j] - offset_x
            pcd_frame.loc[:, j] = pcd_frame.loc[:, j] - offset_x
        for j in [l for l in labels if '(y)' in l]:
            df.loc[frame, j] = df.loc[frame, j] - offset_y
            pcd_frame.loc[:, j] = pcd_frame.loc[:, j] - offset_y

        colors = np.zeros_like(pcd_frame)
        colors = fade_pointclouds(colors)
        colors = colors.reshape(-1, 3)

        pcd_frame = pcd_frame.to_numpy().reshape(-1, 3)
        
        pcd_frame = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(pcd_frame))
        pcd_frame.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd_frame)
    
    return df, pcd_list, keyframes
    

def create_collage(file_name, keyframes,
                   sampling_frequency, og_width, og_height):
    dir_name = os.path.dirname(file_name)
    base_name_without_extension = os.path.basename(file_name).split('.')[0]
    tmp_folder_name = os.path.join(dir_name, '.tmp_' + base_name_without_extension)
    os.mkdir(tmp_folder_name)
    
    collage_frame_width = 425
    collage_frame_height = 500
    top_left_x = int(og_width / 2 - collage_frame_width / 2 + og_width * 0.025)
    top_left_y = int(og_height / 2 - collage_frame_height / 2 - og_height * 0.1125)

    os.system(f'ffmpeg -i {file_name} -filter:v "crop={collage_frame_width}:{collage_frame_height}:{top_left_x}:{top_left_y}" {tmp_folder_name}/cropped.mp4')
    os.system(f'ffmpeg -i {tmp_folder_name}/cropped.mp4 -vf fps={sampling_frequency} {tmp_folder_name}/out%d.png')
    
    ts_top_left_x = int((collage_frame_width / 2) * 0.45)
    ts_top_left_y = int(collage_frame_height * 0.9)        
    for i in range(1, keyframes[-1] + 2):
        if i - 1 not in keyframes:
            # Remove frame
            non_keyframe_file_name = os.path.join(tmp_folder_name, f'out{i}.png')
            os.remove(non_keyframe_file_name)
        else: 
            # Add time stamp to the image
            keyframe_file_name = os.path.join(tmp_folder_name, f'out{i}.png')
            ts_keyframe_file_name = os.path.join(tmp_folder_name, f'out{i}ts.png')
            ts_text =  "%0.2f s" % (i * (1 / sampling_frequency))
            os.system(f'ffmpeg -i {keyframe_file_name} -vf "drawtext=text={ts_text}:fontcolor=black:fontsize=50:x={ts_top_left_x}:y={ts_top_left_y}:" {ts_keyframe_file_name}')

    new_number_count = 1
    for f in keyframes:
        f_number = f + 1
        keyframe_file_name = os.path.join(tmp_folder_name, f'out{f_number}ts.png')
        new_keyframe_file_name = os.path.join(tmp_folder_name, f'out{new_number_count}ts.png')
        os.rename(f'{keyframe_file_name}', f'{new_keyframe_file_name}')
        new_number_count += 1
    os.system(f'ffmpeg -i {tmp_folder_name}/out%dts.png -filter_complex tile=10x1 {dir_name}/{base_name_without_extension}.png')
    
    #shutil.rmtree(tmp_folder_name)


def visualize(df, sampling_frequency, file_name, replace, mode):
    df_viz = df.copy()

    width = 1280
    height = 1024

    if mode == 'collage':
        df_viz, pcd_list, keyframes = prepare_for_static_mode(df)
        render = MoCapViewer(
            sampling_frequency=sampling_frequency, 
            grid_axis = None,
            bg_color = 'white',
            draw_axis = False,
            width = width,
            height = height,
            point_size=4.5
        )
        render.add_point_cloud_animation(point_cloud_list=pcd_list)
    elif mode == 'video':
        render = MoCapViewer(
            sampling_frequency=sampling_frequency,
            grid_color = "lightslategray",
            grid_dimensions = 7,
            bg_color = 'white',
            draw_axis = True,
            width = width,
            height = height
        )
    elif mode == 'inspection':
        # Only differences to video mode are the decreased grid dimensions 
        # and that no file name is required.
        render = MoCapViewer(
            sampling_frequency=sampling_frequency,
            grid_color = "lightslategray",
            grid_dimensions = 3, #5,
            bg_color = 'white',
            draw_axis = True,
            width = width,
            height = height
        )
        '''render = MoCapViewer(
            sampling_frequency=sampling_frequency,
            #grid_color = "lightslategray",
            #grid_dimensions = 5,
            grid_axis=None,
            bg_color = 'white',
            draw_axis = False,
            width = width,
            height = height
        )'''
    else: 
        raise Exception('Mode not supported')

    render.add_skeleton(df_viz, skeleton_connection="vicon", color="red")

    render._MoCapViewer__renderer.GetActiveCamera().SetPosition(-10.0, 0.0, 3.0)
    render._MoCapViewer__renderer.GetActiveCamera().SetViewUp(1.0, 0.0, 0.0)

    #render._MoCapViewer__renderer.GetActiveCamera().SetPosition(0.0, -10.0, 3.0)
    #render._MoCapViewer__renderer.GetActiveCamera().SetPosition(0.0, -10.0, 2.0)
    #render._MoCapViewer__renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)
    #render._MoCapViewer__renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.2)
    #render._MoCapViewer__renderer.GetActiveCamera().Zoom(2.0)


    #render._MoCapViewer__renderer.GetActiveCamera().SetPosition(0.0, -10.0, 3.0)
    #render._MoCapViewer__renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)
    #render._MoCapViewer__renderer.GetActiveCamera().SetViewAngle(10.0)
    



    if not file_name and (mode == 'collage' or mode == 'video'):
        raise Exception(f'Mode {mode} requires a file name')

    if file_name:
        # Setup window to image pipeline
        image_filter = vtk.vtkWindowToImageFilter()
        image_filter.SetInput(render._MoCapViewer__render_window)
        image_filter.SetInputBufferTypeToRGB()
        image_filter.ReadFrontBufferOff()
        image_filter.Update()

        #Setup movie writer
        movie_writer = vtk.vtkOggTheoraWriter()
        if file_name[-4:] != '.mp4':
            file_name += '.mp4'

        if os.path.isfile(file_name) and not replace:
            raise Exception('File already exists. Set replace to true if desired.')

        movie_writer.SetFileName(file_name)
        movie_writer.SetInputConnection(image_filter.GetOutputPort())
        movie_writer.SetRate(sampling_frequency)
        movie_writer.SetQuality(2)
        movie_writer.Start()

        observer_tag = None
        def export_frame(input_1, input_2):
            if render._MoCapViewer__cur_frame >= render._MoCapViewer__max_frames:
                movie_writer.End()
                render._MoCapViewer__render_window_interactor.RemoveObserver(observer_tag)
                # App termination
                render._MoCapViewer__render_window_interactor.TerminateApp()
            else:
                image_filter.Modified()
                movie_writer.Write()

        render._MoCapViewer__render_window_interactor.SetStillUpdateRate(sampling_frequency)
        render._MoCapViewer__render_window_interactor.SetDesiredUpdateRate(sampling_frequency)
        render._MoCapViewer__render_window_interactor.SetNumberOfFlyFrames(sampling_frequency)
        observer_tag = render._MoCapViewer__render_window_interactor.AddObserver('TimerEvent', export_frame)    

        

    render.show_window()
    if file_name:
        movie_writer.End()

    if mode == 'collage':
        time.sleep(5)
        create_collage(file_name, keyframes,
                       sampling_frequency, width, height)


def from_array(arr, sampling_frequency=25, file_name=None, replace=False, mode='collage'):
    if len(arr.shape) != 2:
        arr = arr.reshape(-1, arr.shape[1] * arr.shape[2])
        labels, _ = get_labels()
    df = pd.DataFrame(arr, columns=labels)
    visualize(df, sampling_frequency, file_name, replace, mode)


def from_df(df, sampling_frequency=25, file_name=None, replace=False, mode='collage'):
    labels, _ = get_labels()
    df.columns = labels
    visualize(df, sampling_frequency, file_name, replace, mode)
