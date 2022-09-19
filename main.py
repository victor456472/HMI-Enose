import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtCore import QIODevice, QPoint
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from Design import Ui_MainWindow
import os

class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initGui()
    
    def initGui(self):

        #se crea un objeto de la clase UiMainWindow que contiene los elementos de la interfaz grafica
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #estas declaraciones iniciales permiten programar los botones de la ventana (maximizar, minimizar, restaurar y cerrar)
        #ademas se obtiene la posicion del cursor
        self.ui.boton_normalizar.hide()
        self.click_position=QPoint()
        self.ui.boton_minimizar.clicked.connect(lambda: self.showMinimized())
        self.ui.boton_normalizar.clicked.connect(self.control_normalizar)
        self.ui.boton_maximizar.clicked.connect(self.control_maximizar)
        self.ui.boton_salir.clicked.connect(lambda: self.close())

        #eliminar la barra de titulo y la opacidad
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowOpacity(1)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        #sizeGrip
        self.gripSize=10
        self.grip= QtWidgets.QSizeGrip(self)
        self.grip.resize(self.gripSize, self.gripSize)

        #mover ventana
        self.ui.frame_superior.mouseMoveEvent = self.mover_ventana

        #conexion serial
        self.serial=QSerialPort()
        self.ui.boton_actualizar.clicked.connect(self.read_ports)
        self.ui.boton_conectar.clicked.connect(self.serial_connect)
        self.ui.boton_desconectar.clicked.connect(lambda: self.serial.close())

        #lectura de datos
        self.serial.readyRead.connect(self.read_data)
        self.x = list(np.linspace(0,100,100))
        self.y = list(np.linspace(0,0,100))

        #grafica
        pg.setConfigOption('background', '#2c2c2c')
        pg.setConfigOption('foreground', '#ffffff')
        self.plt = pg.PlotWidget(title='grafica')
        self.ui.grafica.addWidget(self.plt)

        #dataframes
        self.df = pd.DataFrame({
            'ALCOHOL_s1[PPM]':[],
            'MONOXIDO DE CARBONO_S1[PPM]':[],
            'DIHIDROGENO_s1[PPM]':[],
            'ACETONA_s1[PPM]':[],
            'METANO_s1[PPM]':[],
            'ALCOHOL_s2[PPM]':[],
            'MONOXIDO DE CARBONO_S2[PPM]':[],
            'DIHIDROGENO_s2[PPM]':[],
            'ACETONA_s2[PPM]':[],
            'METANO_s2[PPM]':[],
        })

        #gestion de recursos
        self.rawdata_counter=0

        self.read_ports()
    
    def read_ports(self):
        self.baudrates = ['1200', '2400', '4800', '9600', '19200', '38400', '115200']
        portList = []
        ports = QSerialPortInfo().availablePorts()
        for port in ports:
            portList.append(port.portName())
        self.ui.comboBox_puerto.clear()
        self.ui.comboBox_baudrate.clear()
        self.ui.comboBox_puerto.addItems(portList)
        self.ui.comboBox_baudrate.addItems(self.baudrates)
        self.ui.comboBox_baudrate.setCurrentText('9600')
    
    def serial_connect(self):
        self.serial.waitForReadyRead(100)
        self.port = self.ui.comboBox_puerto.currentText()
        self.baud = self.ui.comboBox_baudrate.currentText()
        self.serial.setBaudRate(int(self.baud))
        self.serial.setPortName(self.port)
        self.serial.open(QIODevice.ReadWrite)

    def read_data(self):
        if not self.serial.canReadLine(): return
        rx = self.serial.readLine()
        x=str(rx, 'utf-8').strip()
        x=x.split(',')
        fin=int(x[10])
        #print(f'{x[0]} -- {x[1]} -- {x[2]} -- {x[3]} -- {x[4]} -- {x[5]} -- {x[6]} -- {x[7]} -- {x[8]} -- {x[9]} -- {x[10]}')
        if(fin==0):
            self.y = self.y[1:]
            self.y.append(float(x[0]))
            self.plt.clear()
            self.plt.plot(self.x,self.y,pen=pg.mkPen('#da0037', width=2))
            new_row={
                'ALCOHOL_s1[PPM]':float(x[0]),
                'MONOXIDO DE CARBONO_S1[PPM]':float(x[1]),
                'DIHIDROGENO_s1[PPM]':float(x[2]),
                'ACETONA_s1[PPM]':float(x[3]),
                'METANO_s1[PPM]':float(x[4]),
                'ALCOHOL_s2[PPM]':float(x[5]),
                'MONOXIDO DE CARBONO_S2[PPM]':float(x[6]),
                'DIHIDROGENO_s2[PPM]':float(x[7]),
                'ACETONA_s2[PPM]':float(x[8]),
                'METANO_s2[PPM]':float(x[9]),
            }
            self.df=self.df.append(new_row, ignore_index=True)
            print(self.df)
        elif(fin==1):
            #df2=pd.read_csv('indices/indices.csv')
            #print(df2.shape)
            file_names=os.listdir('datos_recolectados')

            if file_names: #si el directorio esta lleno
                same_name_file=True
                i=0
                while same_name_file:
                    for name in file_names:
                        if(name==f'rawdata{i}.csv'):
                            i=i+1
                        else:
                            same_name_file=False
                self.df.to_csv(f'datos_recolectados/rawdata{i}.csv')
                    
            else: #si el directorio esta vacio
                self.df.to_csv('datos_recolectados/rawdata0.csv')

            self.df = pd.DataFrame({
                'ALCOHOL_s1[PPM]':[],
                'MONOXIDO DE CARBONO_S1[PPM]':[],
                'DIHIDROGENO_s1[PPM]':[],
                'ACETONA_s1[PPM]':[],
                'METANO_s1[PPM]':[],
                'ALCOHOL_s2[PPM]':[],
                'MONOXIDO DE CARBONO_S2[PPM]':[],
                'DIHIDROGENO_s2[PPM]':[],
                'ACETONA_s2[PPM]':[],
                'METANO_s2[PPM]':[],
            })
            
    
    def control_normalizar(self):
        self.showNormal()
        self.ui.boton_normalizar.hide()
        self.ui.boton_maximizar.show()
    
    def control_maximizar(self):
        self.showMaximized()
        self.ui.boton_maximizar.hide()
        self.ui.boton_normalizar.show()

    #tamaño de la ventana
    def resizeEvent(self, event):
        rect = self.rect()
        self.grip.move(rect.right() - self.gripSize, rect.bottom() - self.gripSize) 
    
    #mover la ventana
    def mousePressEvent(self, event):
        self.click_position = event.globalPos()
    
    def mover_ventana(self, event):
        if self.isMaximized()==False:
            if event.buttons() == QtCore.Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.click_position)
                self.click_position = event.globalPos()
                event.accept()
        if event.globalPos().y() <= 5 or event.globalPos().x() <= 5:
            self.showMaximized()
            self.ui.boton_maximizar.hide()
            self.ui.boton_normalizar.show()
        else:
            self.showNormal()
            self.ui.boton_normalizar.hide()
            self.ui.boton_maximizar.show()
    
def main():
    app = QApplication(sys.argv)
    ventana = Application()
    ventana.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()