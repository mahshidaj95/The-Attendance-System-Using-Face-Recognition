import cv2
from PySide2 import QtGui, QtWidgets, QtCore
from PySide2.QtWidgets import QFileDialog, QTableWidgetItem, QListWidgetItem
import os
import ui
import FaceRec
import datetime


class MyQtApp(ui.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__()
        self.setupUi(self)
        self.images_path = []
        self.image_recognizer = None
        self.data = []
        self.selected_image = ''
        self.dir=''

        self.actionAddImage.triggered.connect(self.open_file)
        self.actionDelete_Image.triggered.connect(self.delete_file)
        self.actionLoad.triggered.connect(self.load_images_name)
        self.actionProcess.triggered.connect(self.process)
        self.actionWebcam.triggered.connect(self.webcam)
        self.actionExit.triggered.connect(self.exit)
        self.listWidget.itemDoubleClicked.connect(self.list_action)

    def exit(self):
        self.close()
        QtWidgets.QApplication.quit()

    def open_file(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Image", "Database\\sample",
                                                "Image(*.jpg);;Image(*.png);;", options=options)
        self.selected_image = files[0]
        if files:
            image = QtGui.QImage(files[0])
            pp = QtGui.QPixmap.fromImage(image)
            self.label.setPixmap(pp)
            self.label.show()

    def delete_file(self):
        self.label.clear()

    def load_images_name(self):
        self.dir = QFileDialog.getExistingDirectory(self, 'Select a folder:',
                                               'Database\\dataset',
                                               QFileDialog.ShowDirsOnly)
        images_name_list = [x[2] for x in os.walk(self.dir)]
        if self.dir is not None:
            self.images_path.clear()
            self.listWidget.clear()
        for l in images_name_list[0]:
            self.listWidget.addItem(l.split('.')[0])
            self.images_path.append(self.dir + '/' + l)
        self.image_recognizer = FaceRec.FaceRecognizer(self.images_path)

    def process(self):
        if (self.selected_image is not '') and (self.images_path is not None):
            detected_person_name = self.image_recognizer.ClassifyFace_image(self.selected_image)
            self.set_table(detected_person_name)

    def webcam(self):
        if self.images_path is not None:
            detected_person_name = self.image_recognizer.ClassifyFace_webcam()
            self.set_table(detected_person_name)

    def set_table(self, name):
        if name is not 'Unknown' or '':
            self.data.append((name, str(datetime.datetime.now())))
            self.tableWidget.setRowCount(len(self.data))
            row = 0
            for tup in self.data:
                col = 0
                for item in tup:
                    cellinfo = QTableWidgetItem(item)
                    self.tableWidget.setItem(row, col, cellinfo)
                    col += 1
                row += 1

    def list_action(self, item):
        img_path = self.dir+'/' + item.text() + '.jpg'
        img = cv2.imread(img_path, 1)
        cv2.imshow('Result!', img)



if __name__ == '__main__':
    app = QtWidgets.QApplication()
    qt_app = MyQtApp()
    qt_app.show()
    app.exec_()
