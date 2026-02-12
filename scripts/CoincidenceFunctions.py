import sys
import time
import copy
import queue
import platform
import json
import os
import threading
import statistics
from datetime import datetime, date
from pathlib import Path
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor
from Imaging import Imaging
import re

import tifffile
import cv2
import numpy as np

pc_type = platform.system()
import matplotlib
if pc_type == 'Windows':
    matplotlib.use('Qt5Agg')
elif pc_type == 'Darwin':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


sys.path.append('C:\Program Files\Thermo Scientific Autoscript')
sys.path.append('C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages')

# from autoscript_sdb_microscope_client import SdbMicroscopeClient
# from autoscript_sdb_microscope_client.enumerations import *
# from autoscript_sdb_microscope_client.structures import GetImageSettings
# from autoscript_sdb_microscope_client import SdbMicroscopeClient



class CoincidenceFunctions:
    def __init__(self, oa, on_experiment_stopped=None):
        self.results = None
        self.oa = oa
        if self.oa.tool != 'Arctis':
            raise RuntimeError("This is not the right tool to run the automated tricoincidence routine.")
        self.imaging = Imaging(self.oa)
        self.hfw = 120.0e-6
        self.lock  = threading.Lock()
        self.coin_stop_event = threading.Event()
        self.pause_not_stop = False
        self.data_callback = None
        self.on_experiment_stopped = on_experiment_stopped


#######################################################################################################################
#####       Functions controlling the Arctis
#######################################################################################################################

    def grab_fl_live_image(self, save=True):
        if self.oa.manufacturer != 'Demo':
            
            self.oa.thermo_microscope.imaging.set_active_view(3)

            image = self.oa.thermo_microscope.imaging.get_image()
            if save is True:
                image.save(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
            img = image.data
        else:
            sim = self.fl_data_simulation()
            image = sim()
            if save is True:
                tifffile.imwrite(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"), image)
            img = image
        print("[INFO] Acquiring live fluorescence image.")
        return img

    def grab_reflection_image(self, row=None):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value = CameraFilterType.REFLECTION
            self.oa.thermo_microscope.detector.brightness.value = 0.01
            self.oa.thermo_microscope.detector.camera_settings.binning.value = 4
            self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = 0.001
            emission_color = self.oa.thermo_microscope.detector.camera_settings.emission.type.value
            self.oa.thermo_microscope.detector.camera_settings.emission.start(emission_type=
                                                                              emission_color)
            reflection_image = self.grab_fl_live_image()
            self.oa.thermo_microscope.detector.camera_settings.emission.stop()
            reflection_image.save(os.path.join(self.manual_folder_path, f"reflection_image.tif"))

        else:
            sim = self.fl_data_simulation()
            image = sim()
            tifffile.imwrite(os.path.join(self.manual_folder_path, f"reflection_image.tif"), image)
        print('[INFO] Acquiring reflection image.')

    def grab_fluorescence_image(self, add_on=None, row=None, save=True):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            if self.oa.thermo_microscope.detector.camera_settings.filter.type.value == CameraFilterType.REFLECTION and hasattr(self.manual_binning):
                pass
            else:
                self.manual_binning = self.oa.thermo_microscope.detector.camera_settings.binning.value
                self.manual_brightness = self.oa.thermo_microscope.detector.brightness.value
                self.manual_exposure_time = self.oa.thermo_microscope.detector.camera_settings.exposure_time.value
                self.manual_filter_settings = self.oa.thermo_microscope.detector.camera_settings.filter.type.value
                self.emission_color = self.oa.thermo_microscope.detector.camera_settings.emission.type.value
                print(f"[INFO] The emission color is {self.emission_color}.")
                self.manual_objective_focus =  self.oa.thermo_microscope.detector.camera_settings.focus.value
            image = self.oa.thermo_microscope.imaging.grab_frame()
            if save is True:
                image.save(os.path.join(self.manual_folder_path, f"Fl_image_{add_on}.tif"))
            img = image.data
        else:
            sim = self.fl_data_simulation()
            image = sim()
            tifffile.imwrite(os.path.join(self.manual_folder_path, f"Fl_image_{add_on}.tif"), image)
            img = image
        print('[INFO] Acquiring fluorescence image.')
        return img

    def run_serial_acquisition_fl_images(self, update_callback, stop_event, start_timestamp, row=None, fl_settings=None,
                                         timestamp=None):
        print("[INFO] Serial acquisition started...")
        self.image_queue = queue.Queue(maxsize=2000)
        self.coin_stop_event.clear()
        if self.oa.manufacturer != 'Demo':
            folder_path = Path(os.path.join(self.manual_folder_path, "Images"))
            folder_path.mkdir(parents=True, exist_ok=True)
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = self.manual_exposure_time
            self.oa.thermo_microscope.detector.brightness.value = self.manual_brightness
            self.oa.thermo_microscope.detector.camera_settings.binning.value = self.manual_binning
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value = self.manual_filter_settings
            self.oa.thermo_microscope.imaging.start_acquisition()
            self.oa.thermo_microscope.detector.camera_settings.emission.type.value = self.emission_color
            
            writer_thread = threading.Thread(target=self.image_writer, args=(folder_path, start_timestamp), daemon=True)
            writer_thread.start()
            i = 0
            start_time = datetime.now()
            emission_color = self.oa.thermo_microscope.detector.camera_settings.emission.type.value
            self.oa.thermo_microscope.detector.camera_settings.emission.start(emission_type=
                                                                       emission_color)
            image = self.oa.thermo_microscope.imaging.get_image()
            try:
                if self.oa.thermo_microscope.imaging.state == ImagingState.ACQUIRING:
                    while not stop_event.is_set():
                        now = datetime.now()
                        timestamp = (now - start_time).total_seconds() + start_timestamp
                        new_image = self.oa.thermo_microscope.imaging.get_image()
                        self.image_queue.put((new_image.data.copy(), i))
                        if np.array_equal(new_image.data[:5], image.data[:5]) is False:
                            self.image_queue.put((new_image.data.copy(), i))
                            if update_callback:
                                update_callback(new_image.data, timestamp)
                            image = new_image
                            i += 1
                        else:
                            continue
            except Exception as e:
                print(f"[ERROR] Exception occurred: {e}")
            finally:
                self.oa.thermo_microscope.detector.camera_settings.emission.stop()
                self.oa.thermo_microscope.imaging.stop_acquisition()
                self.image_queue.put(None)
                writer_thread.join()
        else:
            folder_path = Path(os.path.join(self.manual_folder_path, "Images"))
            folder_path.mkdir(parents=True, exist_ok=True)
            
            start_time = datetime.now()
            writer_thread = threading.Thread(target=self.image_writer, args=(folder_path, start_timestamp), daemon=True)
            writer_thread.start()
            i = 0
            sim = self.fl_data_simulation()
            image = sim()
            try:
                while not stop_event.is_set():
                    now = datetime.now()
                    timestamp = (now - start_time).total_seconds() + start_timestamp
                    new_image = sim()
                    if np.array_equal(new_image[:5], image[:5]) is False:
                        self.image_queue.put((new_image.copy(), i))
                        if update_callback:
                            update_callback(new_image, timestamp)
                        image = new_image
                        i += 1
                    else:
                        continue
            except Exception as e:
                print(f"[ERROR] Exception occurred: {e}")
            finally:
                self.image_queue.put(None)
                writer_thread.join()


#######################################################################################################################
#####       Functions required for the execution of the experiment
#######################################################################################################################

    def start_coincidence_milling(self, beam_current, stop_event, resume=False):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(2)
            if self.oa.thermo_microscope.beams.ion_beam.beam_current.value != beam_current:
                self.oa.thermo_microscope.beams.ion_beam.beam_current.value = beam_current
                time.sleep(10)
            if resume is False:
                self.oa.thermo_microscope.patterning.start()
                if self.data_callback:
                    self.data_callback('go', True)
            else:
                self.oa.thermo_microscope.patterning.resume()
            while not stop_event.is_set():
                time.sleep(0.01)
        else:
            print("[INFO] Patterning would start now.")

    def stop_coincidence_milling(self, pause=False):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(2)
            if self.pause_not_stop is False:
                self.oa.thermo_microscope.patterning.stop()
                #self.oa.thermo_microscope.patterning.clear_patterns()
            else:
                self.oa.thermo_microscope.patterning.pause()
        else:
            print('[INFO] Milling stopped!')


    def run_coincidence_experiment(self, callback, stop_event, start_timestamp=0.0, test=False, row=None, beam_current=None):

        def wait_and_finalize_imaging_thread():
            self.imaging_thread.join()
            print("[INFO] Fluorescence experiment stopped.")
            self.stop_coincidence_milling()
            print("[INFO] Milling stopped")

        now = datetime.now()
        if not hasattr(self, "manual_folder_path"):
            self.manual_folder_path = Path(
                os.path.join(self.oa.folder_path, str(date.today()), now.strftime("%H-%M")))
            self.manual_folder_path.mkdir(parents=True, exist_ok=True)
            if self.data_callback:
                self.data_callback('path', self.manual_folder_path)
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            if self.oa.thermo_microscope.detector.camera_settings.filter.type.value == CameraFilterType.REFLECTION:
                pass
            else:
                self.manual_binning = self.oa.thermo_microscope.detector.camera_settings.binning.value
                self.manual_brightness = self.oa.thermo_microscope.detector.brightness.value
                self.manual_exposure_time = self.oa.thermo_microscope.detector.camera_settings.exposure_time.value
                self.manual_filter_settings = self.oa.thermo_microscope.detector.camera_settings.filter.type.value
                self.manual_emission_color = self.oa.thermo_microscope.detector.camera_settings.emission.type
                self.manual_objective_focus = self.oa.thermo_microscope.detector.camera_settings.focus.value

            if self.selected_options['FIB Milling'] is True:
                #beam_current = self.oa.thermo_microscope.beams.ion_beam.beam_current.value
                #self.working_distance = self.oa.thermo_microscope.beams.ion_beam.working_distance.value
                beam_current = self.oa.fib_microscope.get_beam_current(beam_type=BeamType.ION)
#               self.working_distance = self.oa.fib_microscope.get("working_distance", beam_type=BeamType.ION)

                if self.oa.manufacturer != 'Demo':
                    self.oa.thermo_microscope.imaging.set_active_view(2)
                    patterns = self.oa.thermo_microscope.patterning.get_patterns()
                    if len(patterns) == 0:
                        print("[ERROR] Please create a milling pattern!")
                        return

                if start_timestamp == 0.0:
                    if self.selected_options['Before/After FIB Image'] is True:
                        self.imaging.acquire_image(hfw=self.hfw, folder_path=self.manual_folder_path,
                                                beam_type='ion', working_distance=self.working_distance,
                                                autofocus=False, filename=f"FIB-before-image")

                    self.milling_thread = threading.Thread(target=self.start_coincidence_milling, args=(beam_current, stop_event))
                else:
                    resume = True
                    self.milling_thread = threading.Thread(target=self.start_coincidence_milling,
                                                        args=(beam_current, stop_event, resume))
                self.oa.thermo_microscope.beams.ion_beam.beam_current.value = beam_current
                self.milling_thread.start()
                time.sleep(0.5)
                print("[INFO] Starting the milling ...")
                self.imaging_thread = threading.Thread(target=self.run_serial_acquisition_fl_images,
                                                        args=(callback, stop_event, start_timestamp))
                print("[INFO] Performing the fluorescence experiment ...")
                self.imaging_thread.start()


                wait_and_finalize_imaging_thread()
                if self.pause_not_stop is False:
                    self.stop_coincidence_experiment()
            elif self.selected_options['SEM Imaging'] is True:
                print("[INFO] Not yet implemented.")
            

    def stop_coincidence_experiment(self):
        self.oa.thermo_microscope.patterning.clear_patterns()
        if self.selected_options['After Z-Stack'] is True:
            self.acquire_fl_z_stack()
        if self.selected_options['After Reflection Image'] is True:
            self.grab_reflection_image()
        if self.selected_options['Before/After FIB Image'] is True:
            self.imaging.acquire_image(hfw=self.hfw, working_distance=self.working_distance,
                                       folder_path=self.manual_folder_path, beam_type='ion',
                                       autofocus=False, filename=f"FIB-after-image")

    def acquire_fl_z_stack(self, row=None):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value = CameraFilterType.FLUORESCENCE
            self.oa.thermo_microscope.detector.camera_settings.binning.value = self.manual_binning
            self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = self.manual_exposure_time
            self.oa.thermo_microscope.detector.brightness.value = self.manual_brightness
            mid_focus = self.oa.thermo_microscope.detector.camera_settings.focus.value
            z_stack = []
            for i in range(-5, 5):
                print("[INFO] Acquiring Z-Stack. ")
                self.oa.thermo_microscope.detector.camera_settings.focus.value = mid_focus + (i * 250.0e-9)
                image = self.grab_fluorescence_image(save=False)
                if isinstance(image, np.ndarray):
                    z_stack.append(image)
                else:
                    z_stack.append(image.data)
                
                tifffile.imwrite(os.path.join(self.manual_folder_path, "Z_stack_after.tif"), z_stack)
                
            self.oa.thermo_microscope.detector.camera_settings.focus.value = mid_focus
        else:
            z_stack = []
            for i in range(-5, 5):
                image = (np.random.rand(512, 512) * 255).astype(np.uint8)
                if isinstance(image, np.ndarray):
                    z_stack.append(image)
                else:
                    z_stack.append(image.data)

                tifffile.imwrite(os.path.join(self.manual_folder_path, "Z_stack_after.tif"), z_stack)
                

  








