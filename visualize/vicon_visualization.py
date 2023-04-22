#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from PyMoCapViewer import MoCapViewer
from utils.karate import data_info
import vtk

def get_labels(): 
    labels = []
    for l in data_info.joint_to_index.keys():
        labels.append(l + ' (x)')
        labels.append(l + ' (y)')
        labels.append(l + ' (z)')
    return labels

def vizualize(df, sampling_frequency, file_name):
    render = MoCapViewer(sampling_frequency=sampling_frequency)
    render.add_skeleton(df, skeleton_connection="vicon", color="red")
    render._MoCapViewer__renderer.GetActiveCamera().SetPosition(-10.0, 0.0, 3.0)
    render._MoCapViewer__renderer.GetActiveCamera().SetViewUp(1.0, 0.0, 0.0)

    if file_name:
        # Setup window to image pipeline
        imageFilter = vtk.vtkWindowToImageFilter()
        imageFilter.SetInput(render._MoCapViewer__render_window)
        imageFilter.SetInputBufferTypeToRGB()
        imageFilter.ReadFrontBufferOff()
        imageFilter.Update()

        #Setup movie writer
        moviewriter = vtk.vtkOggTheoraWriter() 
        if file_name[-4:] != '.ogv':
            file_name += '.ogv'
        moviewriter.SetFileName(file_name)
        moviewriter.SetInputConnection(imageFilter.GetOutputPort())
        moviewriter.SetRate(sampling_frequency)
        moviewriter.SetQuality(2)
        moviewriter.Start()

        observer_tag = None
        def export_frame(input_1, input_2):
            if render._MoCapViewer__cur_frame >= render._MoCapViewer__max_frames:
                moviewriter.End()
                render._MoCapViewer__render_window_interactor.RemoveObserver(observer_tag)
            else:
                imageFilter.Modified()
                moviewriter.Write()

        render._MoCapViewer__render_window_interactor.SetStillUpdateRate(sampling_frequency)
        render._MoCapViewer__render_window_interactor.SetDesiredUpdateRate(sampling_frequency)
        render._MoCapViewer__render_window_interactor.SetNumberOfFlyFrames(sampling_frequency)
        observer_tag = render._MoCapViewer__render_window_interactor.AddObserver('TimerEvent', export_frame)    
    render.show_window()

def from_array(arr, sampling_frequency=25, file_name=None):
    df = pd.DataFrame(arr, columns=get_labels())
    vizualize(df, sampling_frequency, file_name)

def from_df(df, sampling_frequency=25, file_name=None):
    df.columns = get_labels()
    vizualize(df, sampling_frequency, file_name)
