
import subprocess
import os
from pathlib import Path
from datetime import datetime
from fibsem import utils, structures, microscope
from tkinter import messagebox
import sys
import ast
import yaml
import socket
import numpy as np
from aicsimageio import AICSImage
import xml.etree.ElementTree as ET
from PIL import Image
from PIL.TiffTags import TAGS
from dataclasses import dataclass
import json
import tifffile
import platform
import threading
### HERE THE CORRECT PATH TO AUTOSCRIPT CLIENT MUST BE ADDED
sys.path.append("C:\\Program Files\\Thermo Scientific AutoScript")
sys.path.append("C:\\Program Files\\Enthought\\Python\\envs\\AutoScript\\Lib\\site-packages")
sys.path.append("C:\\Program Files\\Enthought\\Python\\envs\\AutoScript")
sys.path.append("C:\\Program Files\\Enthought\\Python\envs\\AutoScript\\Lib\\site-packages\\autoscript_sdb_microscope_client")
from autoscript_sdb_microscope_client import SdbMicroscopeClient

def error_message(text):
    messagebox.showerror("Error", text)


def create_temp_folder(predefined_path=None):
    """
    This script creates a folder with the current date on the desktop. Inside a temp folder to
    temporarily store data.
    """
    if predefined_path is None:
        current_date = datetime.now().strftime("%Y%m%d")
        desktop_path = Path.home() / "Desktop/"
        folder_path = os.path.join(desktop_path, 'Experiment_Images/', current_date)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    else:
        folder_path = predefined_path
    if not os.path.exists(os.path.join(folder_path + '/Temp')):
        os.makedirs(folder_path + '/Temp')
    return os.path.join(folder_path, 'Temp'), folder_path

class BasicFunctions:
    """
    This class summarizes the basic functions needed as foundation for many other modules.
    It includes the read-in of the textfiles, allows to execute scripts from another
    virtual environment, connects to the microscope.
    """

    def __init__(self, default_settings=False):
        create_temp_folder()
        self.project_root = Path(__file__).resolve().parent.parent
        self.python_root = Path(__file__).resolve().parent.parent.parent.parent
        self.temp_folder_path, self.folder_path = create_temp_folder()
        try:
            self.thermo_microscope = SdbMicroscopeClient()
            self.thermo_microscope.connect()
            self.tool = self.thermo_microscope.service.system.name
            self.manufacturer = 'Thermo'
        except:
            print('Autoscript connection not successful.')
            self.manufacturer = 'Demo'
            self.thermo_microscope = None
            self.tool = 'Arctis'
        self.pc_type = platform.system()
        if self.tool == 'Helios 5 Hydra UX':
            self.tool = 'Hydra'
        # with open(os.path.join(self.project_root, 'scripts', f"czii-stored-stage-positions_{self.tool.lower()}.yaml"), "r") as file:
        #     self.saved_stage_positions = yaml.safe_load(file)
        self.default_settings = default_settings
        self.fib_microscope, self.fib_settings = self.connect_to_microscope()


    def socket_communication(self, target_pc, function, args, role=None):
        if target_pc == 'Meteor_PC':
            SERVER_IP = '10.50.2.119'
            PORT = 23924
            if role is None:
                role = 'server'
        elif target_pc == 'Microscope_PC_Hydra':
            SERVER_IP = ''
            PORT = ''
            if role is None:
                role = 'client'
        elif target_pc == 'Support_PC_Hydra':
            SERVER_IP = ''
            PORT = ''
            if role is None:
                role = 'client'
        else:
            print('Please specify the target PC.')
            return
        command = f"{function} {args}"
        if role == 'client':
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((SERVER_IP, PORT))
                s.sendall(command.encode("utf-8"))

                size = int.from_bytes(s.recv(8), 'big')
                data = b""
                while len(data) < size:
                    chunk = s.recv(min(4096, size - len(data)))
                    if not chunk:
                        break
                    data += chunk
                    import pickle
                    return pickle.loads(data)
        elif role == 'server':
            def handle_client(conn, addr):
                print(f"[SERVER] Connection from {addr}")
                try:
                    command_str = conn.recv(1024).decode("utf-8")
                    print(f"[SERVER] Received command: {command_str}")
                    func_name, *arg_str = command_str.strip().split(maxsplit=1)
                    parsed_args = eval(arg_str[0]) if arg_str else {}

                    if hasattr(self, func_name):
                        method = getattr(self, func_name)
                        result = method(parsed_args) if parsed_args else method()
                    else:
                        result = f"[ERROR] Unknown function '{func_name}'"
                except Exception as e:
                    result = f"[EXCEPTION] {e}"

                data = pickle.dumps(result)
                conn.send(len(data).to_bytes(8, 'big'))
                conn.sendall(data)
                conn.close()
                print("[SERVER] Connection closed.")

            def start_server():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((SERVER_IP, PORT))
                    s.listen()
                    print(f"[SERVER] Listening on {SERVER_IP}:{PORT}...")
                    while True:
                        conn, addr = s.accept()
                        thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                        thread.start()

            # Start server loop in a background thread if needed
            server_thread = threading.Thread(target=start_server, daemon=True)
            server_thread.start()
            print("[SERVER] Server started.")
        else:
            raise ValueError(f"Invalid role: {role}")

    def execute_external_script(self, script, dir_name, parameter=None):
        """
        This function executes a 'script.py' function from a different directory.
        script: script name as str
        dir_name: dir_name as str
        parameter: additional parameters needed by the script as str or list of str
        """
        if self.pc_type == 'Darwin':
            python_path = os.path.join(self.python_root, dir_name, 'venv/bin/python')
        elif self.pc_type == 'Windows':
            python_path = os.path.join(self.python_root, dir_name, 'venv\\Scripts\\python')

        script_path = os.path.join(self.python_root, dir_name, script)

        cmd = [python_path, script_path, '--temp_folder_path', self.temp_folder_path]

        if parameter:
            cmd += ['--parameter'] + parameter
        print("Running command:", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        return result

    def connect_to_microscope(self):
        """
        Establish connection to the microscope.
        manufacturer: 'Demo', 'Thermo', 'Tescan'
        ip: 'localhost', '192.168.0.1' using the default IP for Thermo at the moment.
        tool: 'Hydra', 'Arctis'
        """
        if self.tool == 'Hydra':
            config_path = os.path.join(self.project_root, 'config', 'czii-tfs-hydra-configuration.yaml')
        elif self.tool == 'Arctis':
            config_path = os.path.join(self.project_root, 'config', 'tfs-arctis-configuration.yaml')
        else:
            raise ValueError("No valid tool selected. Options are Hydra and Arctis")

        try:
            if self.default_settings is False:
                fib_microscope, fib_settings = utils.setup_session(manufacturer=self.manufacturer,
                                                                   config_path=config_path)
            else:
                fib_microscope, fib_settings = utils.setup_session(manufacturer=self.manufacturer)
            return fib_microscope, fib_settings

        except Exception as e:
            error_message(f"Connection to microscope failed: {e}")
            sys.exit()

    def read_from_yaml(self, file, imaging_settings_yaml=True):
        """
        User generated yaml file with the basic settings for imaging with the ion/electron beam.
        """
        def calc_constructor(loader, node):
            value = loader.construct_scalar(node)
            return eval(value)

        def looks_like_path(s):
            return "/" in s or "\\" in s or s.endswith(('.yaml', '.yml', '.json', '.txt')) or os.path.isabs(s)


        if looks_like_path(file) is False:
            path = os.path.join(self.project_root, 'scripts', file + '.yaml')
        else:
            path = file

        yaml.add_constructor('!calc', calc_constructor)
        with open(path) as file:
            dictionary = yaml.load(file, Loader=yaml.FullLoader)
        if isinstance(dictionary, dict) and all(not isinstance(v, dict) for v in dictionary.values()):
            if 'scan_rotation' in dictionary:
                dictionary['scan_rotation'] = np.deg2rad(dictionary['scan_rotation'])
            imaging_settings = structures.ImageSettings.from_dict(dictionary)
        elif isinstance(dictionary, dict) and all(isinstance(v, dict) for v in dictionary.values()):
            imaging_settings = {}
            for tool, tool_dict in dictionary.items():
                if 'scan_rotation' in tool_dict:
                    tool_dict['scan_rotation'] = np.deg2rad(tool_dict['scan_rotation'])
                imaging_settings[tool] = structures.ImageSettings.from_dict(tool_dict)
        else:
            raise RuntimeWarning("YAML format is not supported.")

        if imaging_settings_yaml is not True:
             return dictionary
        else:
             imaging_settings = structures.ImageSettings.from_dict(dictionary)
             return imaging_settings, dictionary


    def read_from_dict(self, filename):
        """
        The user should create txt files for his experiment conditions which are then converted to dicts and used
        to set the milling/imaging conditions.
        """
        dictionary = {}
        with open(os.path.join(self.folder_path , filename + '.txt'), 'r') as file:
            for line in file:
                if ":" in line:  # Ensure it's a key-value pair
                    key, value = line.strip().split(":", 1)  # Split on first ":"
                    dictionary[key.strip()] = value.strip()
        keys_convert_to_float = ['milling_current', 'milling_voltage', 'line_integration', 'frame_integration',
                                 'spacing',
                                 'spot_size', 'rate', 'milling_voltage', 'dwell_time', 'hfw', 'voltage',
                                 'working_distance', 'beam_current', 'center_x', 'center_y',
                                 'depth', 'rotation', 'width', 'height', 'passes', 'time']
        keys_convert_to_int = ['frame_integration', 'line_integration']
        keys_convert_to_bool = ['autocontrast', 'autogamma', 'save', 'drift_correction', 'reduced_area',
                                'is_exclusion'
                                'aquire_image']
        keys_convert_to_points = ['stigmation', 'shift']
        str_to_bool = {"true": True, "false": False, "none": None}
        for key in keys_convert_to_bool:
            if key in dictionary:
                dictionary[key] = str_to_bool.get(dictionary[key].lower(), None)
        for key in keys_convert_to_float:
            if key in dictionary:
                if dictionary[key] is not None:
                    dictionary[key] = float(dictionary[key])
        for key in keys_convert_to_points:
            if any(f"{key}{suffix}" in dictionary for suffix in ["X", "Y"]):
                if key == 'Center':
                    dictionary['Point'] = {'x': float(dictionary.pop(f"{key}X")),
                                           'y': float(dictionary.pop(f"{key}Y"))}
                else:
                    dictionary[key] = {'x': float(dictionary.pop(f"{key}X")), 'y': float(dictionary.pop(f"{key}Y"))}
        if 'resolution' in dictionary:
            dictionary['resolution'] = ast.literal_eval(dictionary['resolution'])
        for key in keys_convert_to_int:
            if key in dictionary:
                dictionary[key] = int(dictionary[key])
        if 'scan_rotation' in dictionary:
            dictionary['scan_rotation'] = np.deg2rad(float(dictionary['scan_rotation']))
        return dictionary

    def stage_position_within_limits(self, limit, target_position):
        """
        Verifies that the stage moved as expected and that the current position is reasonably close to the target
        position.
        limit: limit in percent
        target_position: FibsemStagePosition
        """
        current_position = self.fib_microscope.get_stage_position()
        current = [current_position.x, current_position.y, current_position.z, current_position.t, current_position.r]
        target = [target_position.x, target_position.y, target_position.z, target_position.t, target_position.r]
        return all(abs(cur - tar) <= abs(limit/100 * tar) for cur, tar in zip(current, target))

    def import_images(self, path, fl=False):
        img = AICSImage(path)
        if fl:
            image = img.data[0][1]
        else:
            image = img.data[0][0]
            if np.shape(image)[0] > 1:
                image = np.rot90(image, k=2, axes=(1, 2))
        try:
            xml_str = img.metadata
            root = ET.fromstring(xml_str)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            pixels = root.find(".//ome:Pixels", ns)
            pixel_z = float(pixels.attrib.get("PhysicalSizeZ"))
            pixel_y = float(pixels.attrib.get("PhysicalSizeY"))
            pixel_x = float(pixels.attrib.get("PhysicalSizeX"))
        except:
            try:
                ome = img.metadata
                pixel_z_raw = ome.images[0].pixels.physical_size_z
                if pixel_z_raw is None:
                    print("⚠️ physical_size_z is None — defaulting to 1.0 µm")
                    pixel_z = 1.0
                else:
                    pixel_z = float(pixel_z_raw)
                pixel_y = float(ome.images[0].pixels.physical_size_y)
                pixel_x = float(ome.images[0].pixels.physical_size_x)
            except:
                try:
                    image2 = Image.open(path)
                    meta_dict = {TAGS[key]: image2.tag_v2[key] for key in image2.tag_v2}
                    desc_json = json.loads(meta_dict["ImageDescription"])
                    pixel_x = desc_json["pixel_size"]["x"]
                    pixel_y = desc_json["pixel_size"]["y"]
                    pixel_z = 1.0
                except:
                    with tifffile.TiffFile(path) as tif:
                        image2 = tif.asarray()
                        # Access the FEI metadata from tag 34682
                        tag = tif.pages[0].tags.get(34682)
                        if tag is None:
                            raise ValueError("FEI metadata tag (34682) not found in TIFF")

                        # tag.value is already a dict
                        metadata_dict = tag.value

                        # Extract pixel size in nm
                        try:
                            pixel_z = 1.0
                            pixel_x = metadata_dict['Scan']['PixelWidth']*1e6
                            pixel_y = metadata_dict['Scan']['PixelHeight']*1e6
                        except KeyError as e:
                            raise ValueError(f"Pixel size keys not found: {e}")

        return image, (pixel_z, pixel_y, pixel_x)


    def retrieve_stage_position(self, position_name, grid_number=None):
        """
        Imports the stage position from the config file.
        Available positions: sputter, gis, SEM, FIB, FL
        """
        dict_position_names = {'sputter': 'sputter_position',
                               'gis': 'GIS_position',
                                'SEM': 'SEM_topview',
                               'FIB': 'FIB_topview',
                               'FL': 'FL_position'
                                }
        self.tool = 'Hydra'
        if self.tool == 'Hydra':
            if grid_number is None:
                raise RuntimeError("Please select a valid grid.")

            else:
                stage_position = next(
                    (d for d in self.saved_stage_positions if d.get("name") == f"grid{grid_number}_"
                                                                               f"{dict_position_names[position_name]}"), None)
        elif self.tool == 'Arctis':
            stage_position = next(
                (d for d in self.saved_stage_positions if d.get("name") == f"{dict_position_names[position_name]}"),
                None)
        else:
            raise RuntimeError("Stage positions not known for this microscope.")
            sys.exit()

        if stage_position:
            fibsem_stage_position = structures.FibsemStagePosition(x=stage_position['x'],
                                                 y=stage_position['y'],
                                                 z=stage_position['z'],
                                                 r=stage_position['r'],
                                                 t=stage_position['t'])
            return fibsem_stage_position
        else:
            print('Stage position retrieval not valid.')

    def id_available_grids(self):
        if self.manufacturer != 'Demo':
            self.docked_grids = self.thermo_microscope.specimen.autoloader.get_slots(False)
            inventory_done = False
            for grid_slot in self.docked_grids:
                if grid_slot.state != 'Unknown':
                    inventory_done = True
            if inventory_done is not True:
                self.docked_grids = self.thermo_microscope.specimen.autoloader.get_slots(True)
            self.thermo_microscope.autoloader.unload()
            self.available_grids = []
            for grid in self.docked_grids:
                if grid.state == "Occupied":
                    self.available_grids.append(grid)
        else:
            @dataclass
            class Grid:
                id: int
                state: str
            self.available_grids = [Grid(id=1, state='Occupied'),
                                    Grid(id=3, state='Empty'),
                                    Grid(id=4, state='Occupied'),
                                    Grid(id=6, state='Occupied')]

    def autoloader_control(self, new_grid_id=None):
        if self.manufacturer != 'Demo' and self.tool == 'Arctis':
            self.loaded_grid = None
            docked_grids = self.thermo_microscope.specimen.autoloader.get_slots(False)
            for docked_grid in docked_grids:
                for grid in self.available_grids:
                    if docked_grid.id == grid.id and docked_grid.state == 'Empty':
                        self.loaded_grid = grid
                        print(f"The loaded grid is {self.loaded_grid.id}")
                        break
            if new_grid_id is None:
                return
            elif new_grid_id != self.loaded_grid.id or self.loaded_grid is None:
                self.thermo_microscope.specimen.autoloader.load(new_grid_id)

class OverArch(BasicFunctions):
    def __init__(self, default_settings=False):
        super().__init__(default_settings)
    def set_variable(self, name, value):
        setattr(self, name, value)
