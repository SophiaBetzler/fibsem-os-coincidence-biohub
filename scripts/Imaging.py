from sympy import continued_fraction

from Basic_Functions import BasicFunctions, OverArch
from fibsem import structures, acquire, calibration
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import math
import napari
import glob
from skimage import io


class Imaging:
    """
    This class combines all different imaging modalities developed here.
    """
    def __init__(self, oa, beam='ion', imaging_settings = None):
        self.oa = oa
        self.beam = beam
        self.beam_type = getattr(structures.BeamType, self.beam.upper())
        if imaging_settings is None:
            self.imaging_settings, self.imaging_settings_dict = self.oa.read_from_yaml(f"imaging_{self.beam}")
        else:
            self.imaging_settings = imaging_settings
            _, self.imaging_settings_dict = self.oa.read_from_yaml(filename=f"imaging_{self.beam}")
        self.imaging_settings.beam_type = self.beam_type
        self.beam_settings = structures.BeamSettings.from_dict(self.imaging_settings_dict)

    def update_beam_settings(self, new_beam_type):
        """
        The default beam is the ion beam because it is best for CLEM automation. If we want to use the
        electron beam instead, you have to update the imaging settings.
        new_beam_type = 'ion' or 'electron'
        """
        if new_beam_type not in ['electron', 'ion']:
            raise ValueError("Invalid beam type.")
        elif new_beam_type != self.beam:
            imaging_settings, imaging_settings_dict = self.oa.read_from_yaml(filename=f"imaging_{new_beam_type}")
            beam_settings = structures.BeamSettings.from_dict(imaging_settings_dict, self.beam_type)
            return imaging_settings, beam_settings
        else:
            return self.imaging_settings, self.beam_settings

    def acquire_image(self, hfw=None, working_distance=None, plasma_source=None, folder_path=None, save=True,
                      autofocus=False, beam_type=None, filename=None, imshow=False):
        """
        This function connects to the buttons in the GUI. It allows to take electron beam, ion beam and electron and ion
        beam images. TO DO: Add ability to take fluorescence images.
        key = 'electron', 'ion', 'both'
        Data are saved to the hard drive.
        """

        if beam_type is not None and beam_type != self.beam:
            self.imaging_settings, imaging_settings_dict = self.update_beam_settings(beam_type)
            self.beam_settings = structures.BeamSettings.from_dict(self.imaging_settings_dict, self.beam_type)
            self.fib_microscope.set_image

        if filename is None:
            acquisition_time = datetime.now().strftime("%H-%M")
            self.imaging_settings.filename = acquisition_time
        else:
            self.imaging_settings.filename = filename

        if hfw is not None:
            self.imaging_settings.hfw = hfw
        if working_distance is not None:
            self.beam_settings.working_distance = working_distance

        if folder_path is not None:
            self.imaging_settings.path = folder_path
        self.imaging_settings.save = save

        if beam_type == 'ion':
            current_plasma_source = self.oa.fib_microscope.get("plasma_gas")
            if plasma_source is not None and current_plasma_source != plasma_source:
                self.oa.fib_microscope.set("plasma_gas", self.imaging_settings_dict['plasma_source'],
                                    beam_type=self.beam_type)
                self.oa.fib_microscope.set("plasma", self.imaging_settings_dict['plasma'],
                                    beam_type=self.beam_type)

        try:
            self.oa.fib_microscope.set("current", self.beam_settings.beam_current, self.beam_type)
            if autofocus is True and self.oa.fib_settings is not None:
                calibration.auto_focus_beam(self.oa.fib_microscope, self.oa.fib_settings, getattr(structures.BeamType,
                                                                                                  beam_type.upper()))
            image = acquire.new_image(self.oa.fib_microscope, self.imaging_settings)
            if imshow is True:
                plt.ion()  # needed to avoid the QCoreApplication::exec: The event loop is already running error
                plt.imshow(image.data, cmap='gray')
                plt.show()
            return image
        except Exception as e:
            print(f"The image acquisition failed because of {e}.")

    def acquire_multiple(self, dict_parameters, beam_type=None):
        """
        Uses lists of parameters stored in a dictionary to screen multiple imaging conditions.
        The keys of the dictionary must match attributes of the structures.ImageSettings class.
        dict_parameter: dictionary of lists, which can contain the keys 'hfw', 'current', 'voltage', 'stage_bias'
        """
        if beam_type is not None:
            imaging_settings, beam_settings = self.update_beam_settings(beam_type)
        else:
            imaging_settings = self.imaging_settings
            beam_settings = self.beam_settings
        imaging_settings.save = True
        filename = imaging_settings.filename
        for key, values in dict_parameters.items():
            for value in values:
                imaging_settings.key = value
                if key == 'hfw':
                    imaging_settings.hfw = value * 1.0e-6
                if key == 'current':
                    beam_settings.beam_current = value * 1.0e-12
                if key == 'voltage':
                    beam_settings.voltage = value
                if key == 'stage_bias':
                    if self.oa.manufacturer == 'Thermo':
                        self.oa.fib_microscope.connection.detector.custom_settings.mirror_voltage.value = value
                        # self.microscope.connection.detector.custom_settings.suction_tube_voltage.value = value
                    else:
                        print('Cannot set stage bias on the Demo microscope.')
                self.oa.fib_microscope.set_beam_settings(beam_settings)
                now = datetime.now()
                imaging_settings.filename = f"{now:%H-%M-%S}_{key}_{value}"
                imaging_settings.save=True
                acquire.acquire_image(self.oa.fib_microscope, imaging_settings)

    def acquire_tileset(self, method='overview'):
        """
        Acquire a tileset of either the ion-beam or the electron-beam image. A preset magnification and
        image resolution are used.
        """
        if method == 'overview':
            if self.beam != 'ion':
                imaging_settings, beam_settings = self.update_beam_settings('ion')
            else:
                imaging_settings = self.imaging_settings
                beam_settings = self.beam_settings
            imaging_settings = self.imaging_settings
            imaging_settings.hfw = 600e-6
            imaging_settings.resolution = [1048, 1048]
            imaging_settings.save = True
            imaging_settings.path = self.oa.folder_path
            imaging_settings.filename = 'Tiles'
            dx, dy = imaging_settings.hfw, imaging_settings.hfw
            nrows, ncols = 3, 3
            initial_position = self.oa.fib_microscope.get_stage_position()
            for i in range(nrows):
                self.oa.fib_microscope.move_stage_absolute(initial_position)
                self.oa.fib_microscope.stable_move(dx=0, dy=dy * i, beam_type=self.beam_type)
                for j in range(ncols):
                    self.oa.fib_microscope.stable_move(dx=dx, dy=0, beam_type=self.beam_type)
                    imaging_settings.filename = f"tile_{i:03d}_{j:03d}"
                    acquire.new_image(self.oa.fib_microscope, imaging_settings)
                filenames = sorted(glob.glob(os.path.join(imaging_settings.path, "tile*_ib.tif")))
                fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
            for i, fname in enumerate(filenames):
                image = structures.FibsemImage.load(fname)
                ax = axes[-(i // ncols + 1)][i % ncols]
                ax.imshow(image.data, cmap="gray")
                ax.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.001, wspace=0.001)
            plt.savefig(os.path.join(imaging_settings.path, "tiles.png"), dpi=300)
            image = io.imread(os.path.join(imaging_settings.path, "tiles.png"))
            viewer = napari.view_image(image, name='My Image')  # This opens the viewer window
            napari.run()

        # import glob
        # if self.beam == 'electron':
        #     filenames = sorted(glob.glob(os.path.join(imaging_settings.path, "tile*_eb.tif")))
        # elif self.beam == 'ion':
        #     filenames = sorted(glob.glob(os.path.join(imaging_settings.path, "tile*_ib.tif")))

    def fast_acquire(self, number_frames, line_acquisition=False):
        """
        Function which enables fast acquisition using serial readout. The data are stored to numpy dataframes.
        Only works for Autoscript > 4.9.
        number_frames: serial read-out number of frames
        line_acquisition: if set to true, only one line will be acquired, false means full-frame readout
        """
        logging.disable(logging.CRITICAL)
        image_resolution = self.imaging_settings.resolution
        self.imaging_settings.save = False
        self.oa.fib_microscope.connection.beams.electron_beam.scanning.dwell_time.value = self.imaging_settings.dwell_time
        self.oa.fib_microscope.connection.beams.electron_beam.beam_current.value = self.beam_settings.beam_current
        self.oa.fib_microscope.connection.beams.electron_beam.high_voltage.value = self.beam_settings.voltage
        self.oa.fib_microscope.connection.beams.electron_beam.horizontal_field_width.value = self.beam_settings.hfw
        self.oa.fib_microscope.connection.beams.electron_beam.scanning.resolution.value = '1024x884'
        self.oa.fib_microscope.connection.beams.electron_beam.scanning.dwell_time.value = (
            self.imaging_settings.dwell_time)
        if line_acquisition is True:
            self.imaging_settings.reduced_area = structures.FibsemRectangle(top=0.3,
                                                                            left=0.2,
                                                                            width=0.5,
                                                                            height=float(1 / image_resolution[1]))
            # To use fast readout switch to ThermoFisher
            self.oa.fib_microscope.connection.beams.electron_beam.scanning.mode.set_reduced_area(
                self.imaging_settings.reduced_area.top, self.imaging_settings.reduced_area.left,
                self.imaging_settings.reduced_area.width, self.imaging_settings.reduced_area.height)
            image = (self.oa.fib_microscope.connection.imaging.get_image
                     (self.oa.fib_microscope.connection.GetImageSettings(wait_for_frame=True)))
            now = datetime.now()
            ms = now.strftime("%f")[:3]
            array_timeseries = [image.data[0, :]]
            now = datetime.now()
            ms = now.strftime("%f")[:3]
            list_timestamps = [f"{0}_{now:%H-%M-%S}-{ms}"]
            k = 0
            while (self.oa.fib_microscope.connection.microscope.imaging.state ==
                   self.oa.fib_microscope.connection.ImagingState.ACQUIRING and k < number_frames):
                settings = self.oa.fib_microscope.connection.GetImageSettings(wait_for_frame=True)
                image = self.oa.fib_microscope.connection.microscope.imaging.get_image(settings)
                now = datetime.now()
                ms = now.strftime("%f")[:3]
                array_timeseries.append([image.data[0, :]])
                now = datetime.now()
                ms = now.strftime("%f")[:3]
                list_timestamps.append([f"{0}_{now:%H-%M-%S}-{ms}"])
                k=k+1
            np.savetxt(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition.txt", array_timeseries, fmt='%.3f')
            np.save(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition", array_timeseries)
            plt.imshow(array_timeseries, cmap='gray')
            plt.savefig(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition.png", dpi=600, bbox_inches="tight")
            plt.show()
            with open(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition_timestamps.txt", "w") as f:
                for item in list_timestamps:
                    f.write(str(item) + "\n")
        else:
            image = (self.oa.fib_microscope.connection.imaging.get_image
                     (self.oa.fib_microscope.connection.GetImageSettings(wait_for_frame=True)))
            now = datetime.now()
            ms = now.strftime("%f")[:3]
            array_timeseries = [image.data]
            now = datetime.now()
            ms = now.strftime("%f")[:3]
            list_timestamps = [f"{0}_{now:%H-%M-%S}-{ms}"]
            k = 0
            while (self.oa.fib_microscope.connection.microscope.imaging.state ==
                   self.oa.fib_microscope.connection.ImagingState.ACQUIRING and k < number_frames):
                settings = self.oa.fib_microscope.connection.GetImageSettings(wait_for_frame=True)
                image = self.oa.fib_microscope.connection.microscope.imaging.get_image(settings)
                now = datetime.now()
                ms = now.strftime("%f")[:3]
                array_timeseries.append([image.data])
                now = datetime.now()
                ms = now.strftime("%f")[:3]
                list_timestamps.append([f"{0}_{now:%H-%M-%S}-{ms}"])
                k = k + 1
            np.savetxt(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition.txt", array_timeseries, fmt='%.3f')
            np.save(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition", array_timeseries)
            plt.imshow(array_timeseries, cmap='gray')
            plt.savefig(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition.png", dpi=600, bbox_inches="tight")
            plt.show()
            with open(f"{self.oa.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition_timestamps.txt", "w") as f:
                for item in list_timestamps:
                    f.write(str(item) + "\n")
        logging.disable(logging.NOTSET)

