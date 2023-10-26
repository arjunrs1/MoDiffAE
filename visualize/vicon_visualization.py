#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from PyMoCapViewer import MoCapViewer
from utils.karate import data_info
import vtk


def get_labels(): 
    labels = []
    for l in data_info.joint_to_index.keys():
        if 'BACK' in l:
            l = l.replace('BACK', 'RBAK')
        labels.append(l + ' (x)')
        labels.append(l + ' (y)')
        labels.append(l + ' (z)')
    return labels


def visualize(df, sampling_frequency, file_name):
    render = MoCapViewer(sampling_frequency=sampling_frequency)
    render.add_skeleton(df, skeleton_connection="vicon", color="red")
    render._MoCapViewer__renderer.GetActiveCamera().SetPosition(-10.0, 0.0, 3.0)
    render._MoCapViewer__renderer.GetActiveCamera().SetViewUp(1.0, 0.0, 0.0)

    if file_name:
        # Setup window to image pipeline
        image_filter = vtk.vtkWindowToImageFilter()
        image_filter.SetInput(render._MoCapViewer__render_window)
        image_filter.SetInputBufferTypeToRGB()
        image_filter.ReadFrontBufferOff()
        image_filter.Update()

        #Setup movie writer
        movie_writer = vtk.vtkOggTheoraWriter()
        if file_name[-4:] != '.ogv':
            file_name += '.ogv'
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
            else:
                image_filter.Modified()
                movie_writer.Write()

        render._MoCapViewer__render_window_interactor.SetStillUpdateRate(sampling_frequency)
        render._MoCapViewer__render_window_interactor.SetDesiredUpdateRate(sampling_frequency)
        render._MoCapViewer__render_window_interactor.SetNumberOfFlyFrames(sampling_frequency)
        observer_tag = render._MoCapViewer__render_window_interactor.AddObserver('TimerEvent', export_frame)    
    render.show_window()


def from_array(arr, sampling_frequency=25, file_name=None):
    if len(arr.shape) != 2:
        arr = arr.reshape(-1, arr.shape[1] * arr.shape[2])
    df = pd.DataFrame(arr, columns=get_labels())
    visualize(df, sampling_frequency, file_name)


def from_df(df, sampling_frequency=25, file_name=None):
    df.columns = get_labels()
    visualize(df, sampling_frequency, file_name)
