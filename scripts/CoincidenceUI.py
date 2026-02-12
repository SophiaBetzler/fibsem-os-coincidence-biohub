from Basic_Functions import OverArch
from CoincidenceFunctions import *
import platform
import time
import numpy as np
import sys
import json
import os
import tifffile
import cv2
import threading
import statistics
import matplotlib
pc_type = platform.system()
if pc_type == 'Windows':
    matplotlib.use('Qt5Agg')
elif pc_type == 'Darwin':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RectangleSelector, Button

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QFormLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox, QCheckBox, QGridLayout, QFileDialog,
    QDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
)
from PyQt5.QtCore import Qt, QEventLoop, QObject, pyqtSignal, QThread, QTimer, QMetaObject, QRectF, QPointF
from PyQt5.QtGui import QBrush, QColor, QIcon, QImage, QPixmap, QPen
from multiprocessing import Queue


########################################################################################################################
### Class called to start the GUIs for either manual or automatic coincidence experiments  #############################
########################################################################################################################

class CoincidenceGUI:
    """
    Sole purpose of this class is to open the GUI controlling the process.
    """
    def __init__(self, oa):
        self.oa = oa(default_settings=True)
        self.coincidence = CoincidenceFunctions(oa=self.oa)
        self.app = QApplication(sys.argv)
        self.run()

    def run(self):
        manual_win = QMainWindow()
        manual_widget = self.manual_gui_window
        manual_win.setCentralWidget(manual_widget)
        manual_win.setWindowTitle('Tool to perform manual coincidence experiment on the Arctis.')
        manual_win.resize(1200, 800)
        manual_win.show()
        self.app.exec()


    def error_messagebox(self, text):
        box = QMessageBox()
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Warning")
        box.setText(text)
        box.setStandardButtons(QMessageBox.Abort)
        choice = box.exec_()
        if choice == QMessageBox.Abort:
            return

    def info_messagebox(self, text, title):
        box = QMessageBox()
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QMessageBox.Abort)
        choice = box.exec_()
        if choice == QMessageBox.Ok:
            return

########################################################################################################################
### GUI Tools for the manual coincidence experiment ####################################################################
########################################################################################################################


### Subclass: ROI selection tool #################################################################################################
HANDLE_SIZE = 5
class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, rect):
        super().__init__(rect)
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setBrush(QColor(255, 0, 0, 50))
        self.setPen(QPen(Qt.red, 2))
        self.handles = {}
        self.handle_size = HANDLE_SIZE
        self.handle_selected = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None

    def boundingRect(self):
        o = self.handle_size / 2
        return self.rect().adjusted(-o, -o, o, o)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        self.updateHandlesPos()
        painter.setBrush(QColor(0, 255, 0))
        for handle, rect in self.handles.items():
            painter.drawRect(rect)

    def updateHandlesPos(self):
        s = self.handle_size
        b = self.rect()
        self.handles['tl'] = QRectF(b.left()-s/2, b.top()-s/2, s, s)
        self.handles['tr'] = QRectF(b.right()-s/2, b.top()-s/2, s, s)
        self.handles['bl'] = QRectF(b.left()-s/2, b.bottom()-s/2, s, s)
        self.handles['br'] = QRectF(b.right()-s/2, b.bottom()-s/2, s, s)

    def mousePressEvent(self, event):
        for k, r in self.handles.items():
            if r.contains(event.pos()):
                self.handle_selected = k
                break
        self.mouse_press_pos = event.pos()
        self.mouse_press_rect = self.rect()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.handle_selected is not None:
            diff = event.pos() - self.mouse_press_pos
            r = QRectF(self.mouse_press_rect)
            if self.handle_selected == 'tl':
                r.setTopLeft(r.topLeft() + diff)
            elif self.handle_selected == 'tr':
                r.setTopRight(r.topRight() + diff)
            elif self.handle_selected == 'bl':
                r.setBottomLeft(r.bottomLeft() + diff)
            elif self.handle_selected == 'br':
                r.setBottomRight(r.bottomRight() + diff)
            self.setRect(r)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.handle_selected = None
        super().mouseReleaseEvent(event)

### 2. Subclass: LiveViewer #########################################################################################################

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, with_roi=False):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setPos(0, 0)
        self.scene.addItem(self.image_item)
        self.roi = None
        if with_roi:
            self.roi = ResizableRectItem(QRectF(100, 100, 120, 80))
            self.scene.addItem(self.roi)
        self.image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        self.vmin, self.vmax = 0, 255
        self.setFixedSize(520, 520)
        self.scale_factor = 1.15

    def update_image(self, new_image, crop=False, display_scale=0.25):
        self.image = new_image

        if crop:
            roi_rect = self.get_roi_rect()

            if roi_rect is None:
                h, w = new_image.shape
                cx, cy = w//2, h//2
            else:
                cx = int(roi_rect.x() + roi_rect.width()/2)
                cy = int(roi_rect.y() + roi_rect.height()/2)
            half_size=256
            x_start = max(cx - half_size, 0)
            y_start = max(cy - half_size, 0)
            x_end = min(cx +  half_size, new_image.shape[1])
            y_end = min(cy + half_size, new_image.shape[0])
            new_image = new_image[y_start:y_end, x_start:x_end]

        img = np.clip((new_image - self.vmin) / (self.vmax - self.vmin) * 255, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_item.setPixmap(pixmap)
        self.image_item.update()
        self.scene.update()
        self.viewport().update()


    def get_roi_rect(self):
        if self.roi is None:
            return None

        # Get ROI's rectangle in scene coordinates
        roi_rect_in_scene = self.roi.mapRectToScene(self.roi.rect())

        # Get image item's top-left scene position
        image_top_left = self.image_item.scenePos()

        # Offset ROI position relative to image
        x = roi_rect_in_scene.left() - image_top_left.x()
        y = roi_rect_in_scene.top() - image_top_left.y()
        w = roi_rect_in_scene.width()
        h = roi_rect_in_scene.height()
        return QRectF(x, y, w, h)

    def get_roi_data(self):
        roi = self.get_roi_rect()
        if roi is None:
            return np.array([])
        x, y, w, h = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
        return self.image[y:y+h, x:x+w] if w > 0 and h > 0 else np.array([])

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1 / self.scale_factor, 1 / self.scale_factor)


### Main GUI window which relies on the other two classes as helpers ###################################################

class ManualCoincidenceGUI(QWidget):
    image_received = pyqtSignal(object, float)
    def __init__(self, oa, coin):
        super().__init__()
        self.oa = oa
        self.coincidence = coin
        self.static_view = ZoomableGraphicsView(with_roi=True)
        self.dynamic_view = ZoomableGraphicsView(with_roi=False)
        self.image_received.connect(self.update_dynamic_image)
        self.coincidence.set_data_callback(self.receive_data)
        self.x_data = []
        self.y_data = []
        self.experiment_start = 0
        self.go = False
        self.done = False
        self.start_timestamp = 0.0
        self.t0 = None

        ### Figure layout ###
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.static_view)
        image_layout.addWidget(self.dynamic_view)

        ### Plot_Histogram layout ###
        plot_hist_layout = QHBoxLayout()

        # Buttons for Histogram Control
        self.vmin, self.vmax = 0, 1500
        self.hist_min_input = QLineEdit("0")
        self.hist_max_input = QLineEdit("1500")
        self.hist_min_input.setText(str(self.vmin))
        self.hist_max_input.setText(str(self.vmax))
        self.hist_update_button = QPushButton("Update Display Range")
        self.hist_update_button.clicked.connect(self.update_display_range)

        hist_button_layout = QGridLayout()
        hist_button_layout.addWidget(QLabel("Min:"), 0, 0)
        hist_button_layout.addWidget(self.hist_min_input, 0, 1)
        hist_button_layout.addWidget(QLabel("Max:"), 1, 0)
        hist_button_layout.addWidget(self.hist_max_input, 1, 1)
        hist_button_layout.addWidget(self.hist_update_button, 2, 0, 1, 2, alignment=Qt.AlignCenter)
        hist_button_widget = QWidget()
        hist_button_widget.setLayout(hist_button_layout)

        # Plots
        plots_layout = QVBoxLayout()
        self.plots = plt.figure(figsize=(15, 15), dpi=150, constrained_layout=True)
        self.plot_canvas = FigureCanvas(self.plots)
        gs = GridSpec(1, 10, figure=self.plots)
        self.ax_line = self.plots.add_subplot(gs[0, 1:6])
        self.ax_line.set_xlabel("Time (s)", fontsize=6)
        self.ax_line.set_ylabel("Mean Intensity", fontsize=6)
        self.ax_line.tick_params(axis='both', labelsize=6)
        self.ax_hist = self.plots.add_subplot(gs[0, 7:10])
        self.ax_hist.tick_params(axis='both', labelsize=6)
        self.ax_hist.set_xlim(self.vmin, self.vmax)
        plots_layout.addWidget(self.plot_canvas)
        plots_layout.setContentsMargins(10, 0, 10, 0)
        plots_widget = QWidget()
        plots_widget.setLayout(plots_layout)

        plot_hist_layout.addWidget(plots_widget)
        plot_hist_layout.addWidget(hist_button_widget)
        plot_hist_container = QWidget()
        plot_hist_container.setLayout(plot_hist_layout)
        plot_hist_container.setMinimumHeight(300)

        ### Checkbox layout ###
        options_defaults = {
            "FIB Milling": True,
            "SEM Imaging": False,
            "Before/After FIB Image": True,
            "After Z-Stack": True,
            "After Reflection Image": False,
        }
        self.coincidence.selected_options = {}
        self.checkbox_layout = QHBoxLayout()
        for label, default_checked_state in options_defaults.items():
            checkbox = QCheckBox(label)
            checkbox.setChecked(default_checked_state)
            checkbox.stateChanged.connect(self.make_checkbox_callback(label))
            self.checkbox_layout.addWidget(checkbox)
            self.coincidence.selected_options[label] = default_checked_state

        ### Button layout ###
        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.update_static_image)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_coincidence_experiment)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_coincidence_experiment)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_coincidence_experiment)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_coincidence_experiment)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.resume_button)
        button_layout.addWidget(self.stop_button)

        ### Main layout ###
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addWidget(plot_hist_container)
        main_layout.addLayout(self.checkbox_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self.dynamic_img = np.zeros((512, 512), dtype=np.uint8)
        self.static_img = self.coincidence.grab_fl_live_image()
        self.update_display_range()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dynamic_image)

    ####################################################
    ### Functions to update the images and live plot ###
    ####################################################

    def update_static_image(self):
        image = self.coincidence.grab_fl_live_image()
        self.static_view.update_image(image)
        self.static_img = image
        self.static_view.update_image(image)
        return image

    def update_dynamic_image(self, image=None, timestamp=None):
        if image is not None:
            self.dynamic_img = image
        if not hasattr(self, "dynamic_img"):
            return
        self.dynamic_view.update_image(self.dynamic_img)
        self.update_plot()



    def update_plot(self):

        if not hasattr(self, "_line_plot"):
            self._line_plot, = self.ax_line.plot([], [], label="Mean ROI Intensity", color='blue')
            self.ax_line.set_xlabel("Time (s)", fontsize=6)
            self.ax_line.set_ylabel("Mean Intensity", fontsize=6)
            self.ax_line.tick_params(axis='both', labelsize=6)
        if not hasattr(self, "_histogram"):
            self._histogram = self.ax_hist.hist([], bins=50, color='gray')[2]
            self.ax_hist.set_xlim(self.vmin, self.vmax)

        roi = self.static_view.get_roi_rect()
        if roi:
            x, y, w, h = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
            roi_data = self.dynamic_view.image[y:y + h, x:x + w] if w > 0 and h > 0 else np.array([])
            if roi_data.size > 0:
                mean_intensity = roi_data.mean()
                self.x_data.append(self.timestamp)
                self.y_data.append(mean_intensity)
        self.ax_line.set_xlim(0, self.timestamp)
        self._line_plot.set_data(self.x_data, self.y_data)
        self.ax_line.relim()
        self.ax_line.autoscale_view()

        for patch in self._histogram:
            patch.remove()
        self._histogram = self.ax_hist.hist(self.dynamic_view.image.flatten(), bins=50, color='gray')[2]
        self.ax_hist.set_xlim(self.vmin, self.vmax)

        self.plot_canvas.draw()


    def update_display_range(self):
        try:
            self.vmin = float(self.hist_min_input.text())
            self.vmax = float(self.hist_max_input.text())
            self.static_view.vmin = self.vmin
            self.dynamic_view.vmin = self.vmin
            self.static_view.vmax = self.vmax
            self.dynamic_view.vmax = self.vmax
            self.dynamic_view.update_image(self.dynamic_img)
            self.static_view.update_image(self.static_img)
        except ValueError:
            pass

    ####################################################
    ### Functions enacted by the button clicks       ###
    ####################################################

    def start_coincidence_experiment(self, start_timestamp=0.0):
        if self.coincidence.coin_stop_event.is_set() is True:
            print('[ERROR] Experiment already running.')
            return

        self.coincidence.coin_stop_event.clear()

        def gui_safe_callback(image, timestamp):
            def safe_update():
                self.update_dynamic_image(image, timestamp)

            self.image_received.emit(image, timestamp)
            self.timestamp = timestamp

        def run():
            self.coincidence.run_coincidence_experiment(callback=gui_safe_callback,
                                                    stop_event=self.coincidence.coin_stop_event,
                                                        start_timestamp=start_timestamp)

        threading.Thread(target=run, daemon=True).start()
        if self.start_timestamp == 0.0:
            self.experiment_start = time.time()
        if self.go is True:
            self.timer.start(100)

    def pause_coincidence_experiment(self):
        if self.coincidence.coin_stop_event.is_set() is True:
            print("[ERROR] Please start the experiment before stopping it.")
            return
        self.coincidence.coin_stop_event.set()
        self.timer.stop()
        self.coincidence.pause_not_stop = True
        print("[INFO] Execution of the coincidence experiment paused.")

    def resume_coincidence_experiment(self):
        self.timer.start()
        print(self.coincidence.coin_stop_event.is_set())
        if self.coincidence.coin_stop_event.is_set() is not True:
            print("[ERROR] Please pause the experiment before trying to resume.")
            return
        self.coincidence.coin_stop_event.clear()
        start_timestamp = self.timestamp
        self.start_coincidence_experiment(start_timestamp=start_timestamp)
        print("[INFO] Execution of the coincidence experiment resumed ...")

    def clear_line_plot(self):
        self.x_data.clear()
        self.y_data.clear()
        if hasattr(self, "_line_plot"):
            self._line_plot.set_data([], [])
        self.ax_line.relim()
        self.ax_line.autoscale_view()
        self.plot_canvas.draw_idle()


    def stop_coincidence_experiment(self):
        if self.coincidence.coin_stop_event.is_set() is True:
            print('[ERROR] Please start the experiment before stopping it.')
            return
        self.timer.stop()
        self.coincidence.pause_not_stop = False
        self.coincidence.coin_stop_event.set()
        self.dynamic_img = np.zeros_like(self.dynamic_img)
        self.dynamic_view.update_image(self.dynamic_img)
        roi = self.static_view.get_roi_rect()
        roi_coordinates = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
        self.write_data_to_disk(roi_coordinates, self.x_data, self.y_data)
        self.coincidence.coin_stop_event.clear()
        self.clear_line_plot()
        self.start_timestamp = 0.0

    def write_data_to_disk(self, roi_coordinates, x_data, y_data):
        before_img = cv2.imread(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
        cv2.rectangle(before_img, (roi_coordinates[0], roi_coordinates[1]),
                      (roi_coordinates[0] + roi_coordinates[2], roi_coordinates[1] + roi_coordinates[3]),
                      color=(0, 0, 255), thickness=5)
        cv2.imwrite(os.path.join(self.path, "ROI_after_exp.tif"), before_img)
        formatted_times = [
            f"{int((x - x_data[0]) // 60):02}:{int((x - x_data[0]) % 60):02}.{int(((x - x_data[0]) % 1) * 1000):03}"
            for x in x_data]
        intensity_data = np.column_stack((formatted_times, y_data))
        np.savetxt(os.path.join(self.path, "roi_intensities.csv"), intensity_data, delimiter=",", fmt="%s",
                   header="timestamp, average_intensity", comments='')
        np.save(os.path.join(self.path, "roi_intensities.npy"), intensity_data)

        print("[INFO] Coincidence experiment terminated successfully.")

    def show_experiment_finished_message(self):
        box = QMessageBox()
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle('Success!')
        box.setText('The coincidence experiment completed successfully and all data was written to the disk.\n\n'
                    'Thank you for using this tool!\n\n'
                    'Have a great day!')
        box.setStandardButtons(QMessageBox.Ok)
        choice = box.exec_()
        if choice == QMessageBox.Ok:
            return

    def write_final_data_to_disk(self, roi_coordinates, x_data, y_data):
        before_img = cv2.imread(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
        cv2.rectangle(before_img, (roi_coordinates[0], roi_coordinates[1]),
                      (roi_coordinates[0] + roi_coordinates[2], roi_coordinates[1] + roi_coordinates[3]),
                      color=(0, 0, 255), thickness=5)
        cv2.imwrite(os.path.join(self.path, "ROI_after_exp.tif"), before_img)
        formatted_times = [
            f"{int((x - x_data[0]) // 60):02}:{int((x - x_data[0]) % 60):02}.{int(((x - x_data[0]) % 1) * 1000):03}"
            for x in x_data]
        intensity_data = np.column_stack((formatted_times, y_data))
        np.savetxt(os.path.join(self.path, "roi_intensities.csv"), intensity_data, delimiter=",", fmt="%s",
                   header="timestamp, average_intensity", comments='')
        np.save(os.path.join(self.path, "roi_intensities.npy"), intensity_data)

        if self.done is True:
            self.show_experiment_finished_message()

        print("[INFO] Coincidence experiment terminated successfully.")

    ####################################################
    ### Functions enabling communication             ###
    ####################################################
    def make_checkbox_callback(self, key):
        def callback(state):
            is_checked = (state == Qt.Checked)
            self.coincidence.selected_options[key] = is_checked

            # Enforce exclusivity between FIB Milling and SEM Imaging
            if is_checked and key in ["FIB Milling", "SEM Imaging"]:
                other_key = "SEM Imaging" if key == "FIB Milling" else "FIB Milling"
                self.coincidence.selected_options[other_key] = False

                # Find and uncheck the other checkbox in the layout
                for i in range(self.checkbox_layout.count()):
                    widget = self.checkbox_layout.itemAt(i).widget()
                    if isinstance(widget, QCheckBox) and widget.text() == other_key:
                        widget.blockSignals(True)  # Avoid triggering the callback again
                        widget.setChecked(False)
                        widget.blockSignals(False)

        return callback


    def receive_data(self, data_type, data):
        def receive_path(data):
            self.path = data

        def receive_go(data):
            self.go = data

        def receive_done(data):
            self.done = data

        if data_type == "path":
            receive_path(data)
        elif data_type == 'go':
            receive_go(data)
        elif data_type == 'done':
            receive_done(data)


#######################################################################################################################
#####       Functions which open separate plotting windows
#######################################################################################################################

    def define_roi(self):
        """
        Define a ROI in the fluorescence image which will be used to calculate the average.
        This script will take the currently displayed image in the 3 view of XT as reference.
        """
        roi_coords = [None]  # Use a mutable object to capture updates

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            roi_coords[0] = (xmin, xmax, ymin, ymax)

        def on_ok_clicked(event):
            if roi_coords[0] is not None:
                print(
                    f"Final ROI confirmed: x={roi_coords[0][2]}:{roi_coords[0][3]}, y={roi_coords[0][0]}:{roi_coords[0][1]}")
                plt.close(fig)
                if self.event_loop:
                    self.event_loop.quit()
            else:
                print("No ROI selected yet.")

        image = tifffile.imread(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(image.data, cmap='gray')
        ax.set_title("Draw ROI, then click OK to confirm")


        self.selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)

        ok_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.ok_button = Button(ok_ax, 'OK')
        self.ok_button.on_clicked(on_ok_clicked)
        plt.show(block=False)
        self.event_loop = QEventLoop()
        self.event_loop.exec_()

        file_path = os.path.join(self.oa.temp_folder_path, f"fl_image.tif")

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print("File does not exist.")

        if roi_coords:
            #plt.imshow(image[roi_coords[0][2]:roi_coords[0][3], roi_coords[0][0]:roi_coords[0][1]])
            #plt.show()
            return (roi_coords[0][2],roi_coords[0][3], roi_coords[0][0],roi_coords[0][1])
        else:
            print("No ROI was selected.")
            return None

    def test_experiment(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.error_messagebox("Please select a sample position to run the test experiment.")
            return

        row = selected[0].row()
        dialog = TestExperimentDialog(self.coincidence, row, self.oa, self)
        dialog.exec_()
        self.update_status(row=row, status="TEST")


    def update_status(self, row: int, status: str):
        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignCenter)

        if status == "OK":
            item.setText("✓")
            item.setForeground(QBrush(QColor("green")))
        elif status == "FAIL":
            item.setText("✗")
            item.setForeground(QBrush(QColor("red")))
        elif status == "TEST":
            item.setText("Test")
            item.setForeground(QBrush(QColor("blue")))
        else:
            item.setText("?")
            item.setForeground(QBrush(QColor("gray")))

        self.table.setItem(row, 4, item)


