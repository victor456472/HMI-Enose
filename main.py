import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtCore import QIODevice, QPoint
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
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
        self.ui.boton_desconectar.hide()
        self.ui.boton_desconectar.clicked.connect(self.serial_disconnect)

        #lectura de datos
        self.serial.readyRead.connect(self.read_data)
        self.x = list(np.linspace(0,300,300))
        self.y = list(np.linspace(0,0,300))
        self.y1 = list(np.linspace(0,0,300))
        self.y2 = list(np.linspace(0,0,300))
        self.y3 = list(np.linspace(0,0,300))
        self.y4 = list(np.linspace(0,0,300))
        self.y5 = list(np.linspace(0,0,300))
        self.y6 = list(np.linspace(0,0,300))
        self.y7 = list(np.linspace(0,0,300))
        self.y8 = list(np.linspace(0,0,300))
        self.y9 = list(np.linspace(0,0,300))

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
        self.df2 =pd.DataFrame({
            'identifier':[],
            'prom_alcohol_s1[PPM]':[],
            'max_val_alcohol_s1[PPM]':[],
            'prom_alcohol_s2[PPM]':[],
            'max_val_alcohol_s2[PPM]':[],
            'prom_metano_s1[PPM]':[],
            'max_val_metano_s1[PPM]':[],
            'prom_metano_s2[PPM]':[],
            'max_val_metano_s2[PPM]':[],
            'razon_max_value_metano_alcohol_s1':[],
            'razon_max_value_metano_alcohol_s2':[],
            'tamaño[cm]':[],
            'categoria':[]
        })

        #generar y borrar datos
        self.deshabilitar_generar_datos()
        self.deshabilitar_borrar_muestra()
        self.ui.boton_generar_datos.clicked.connect(self.calcular_values_dataframe)
        self.ui.boton_borrar_muestra.clicked.connect(self.borrar_muestra)

        #gestion de recursos
        self.rawdata_counter=0
        self.habilitar_data_save=False
        self.categorias=['1', '2', '3', '4', '5', '6', '7']
        self.borrar_generar_datos = False
        self.sc=StandardScaler()
        self.MLP_classifier=MLPClassifier(hidden_layer_sizes=(150,100,50), 
                                          max_iter=300,
                                          activation = 'relu',
                                          solver='adam',
                                          random_state=1)
        self.infLimit=300
        self.supLimit=self.infLimit*2

        #machine learning buttons and legends
        self.deshabilitar_clasificar()
        self.deshabilitar_entrenar()
        self.apagar_titulo_clasificar()
        self.ui.boton_clasificar.clicked.connect(self.clasificar)
        self.ui.boton_entrenar.clicked.connect(self.entrenar)
        self.door1=True

        #panel de monitoreo
        self.ui.check_alcohol_s1.setChecked(True)
        self.ui.check_alcohol_s1.clicked.connect(self.actualizarGraficas)
        self.ui.check_alcohol_s2.clicked.connect(self.actualizarGraficas)
        self.ui.check_acetona_s1.clicked.connect(self.actualizarGraficas)
        self.ui.check_acetona_s2.clicked.connect(self.actualizarGraficas)
        self.ui.check_co_s1.clicked.connect(self.actualizarGraficas)
        self.ui.check_co_s2.clicked.connect(self.actualizarGraficas)
        self.ui.check_dihidrogeno_s1.clicked.connect(self.actualizarGraficas)
        self.ui.check_dihidrogeno_s2.clicked.connect(self.actualizarGraficas)
        self.ui.check_metano_s1.clicked.connect(self.actualizarGraficas)
        self.ui.check_metano_s2.clicked.connect(self.actualizarGraficas)

        #boton de ajuste ambiental
        self.ui.ambientAdjust.setEnabled(False)
        self.ui.ambientAdjust.clicked.connect(self.ajusteAmbiental)
        self.ambienteAjustado=False
        self.lastOffset=0
        self.lastOffset2=0
        self.offset=0 #32
        self.offset2=0 #3400
        self.door0=True

        #entrada manual de datos
        self.ui.comboBox_categoria.addItems(self.categorias)
        self.ui.comboBox_categoria.setCurrentText('1')

        self.read_ports()

    def changeIconAdjusted(self):
        self.ui.ambientAdjust.setStyleSheet("QPushButton{"
                                            "image:url(:/images/iconos/Imagen13.png);"
                                            "}"
                                            "QPushButton:hover{"
                                            "image:url(:/images/iconos/Imagen12.png);"
                                            "}")

    def changeIconNoAdjusted(self):
        self.ui.ambientAdjust.setStyleSheet("QPushButton{"
                                            "image:url(:/images/iconos/Imagen12.png);"
                                            "}"
                                            "QPushButton:hover{"
                                            "image:url(:/images/iconos/Imagen13.png);"
                                            "}")
    
    def deshabilitarAjusteAmbiental(self):
        self.ui.ambientAdjust.setStyleSheet("QPushButton{"
                                            "image:url(:/images/iconos/Imagen14.png);"
                                            "}")
        
        self.ui.ambientAdjust.setEnabled(False)

    def verifyAndChangeIconAdj(self, mux=False):
        if (self.ambienteAjustado ^ mux):
            self.changeIconAdjusted()
        else:
            self.changeIconNoAdjusted()

    def habilitarAjusteAmbiental(self):
        self.ui.ambientAdjust.setEnabled(True)
        if self.door0:
            self.verifyAndChangeIconAdj()
            self.door0=False
        else:
            pass

    def ajusteAmbiental(self):
        if self.ambienteAjustado:
            self.ambienteAjustado=False
            self.offset=0
            self.offset2=0
            self.lastOffset=-self.lastOffset
            self.lastOffset2=-self.lastOffset2
            self.changeIconNoAdjusted()
        else:
            self.offset=self.df.loc[self.df.index[-1],"ALCOHOL_s1[PPM]"]-10
            self.offset2=self.df.loc[self.df.index[-1],"METANO_s1[PPM]"]-1300
            self.lastOffset=self.offset
            self.lastOffset2=self.offset2
            self.ambienteAjustado=True
            self.changeIconAdjusted()
        
        for element in range(0,len(self.y)):
            if self.y[element]==0:
                pass
            else:
                print(f'{self.y[element]}-{self.lastOffset}={self.y[element]-self.lastOffset}')
                self.y[element]=self.y[element]-self.lastOffset

        for element in range(0,len(self.y4)):
            if self.y4[element]==0:
                pass
            else:
                print(f'{self.y4[element]}-{self.lastOffset2}={self.y4[element]-self.lastOffset2}')
                self.y4[element]=self.y4[element]-self.lastOffset2

        self.df["ALCOHOL_s1[PPM]"]=self.df["ALCOHOL_s1[PPM]"]-self.lastOffset
        self.df["METANO_s1[PPM]"]=self.df["METANO_s1[PPM]"]-self.lastOffset2
                  
        self.actualizarGraficas()

    def actualizarGraficas(self):
        self.plt.clear()
        if(self.ui.check_alcohol_s1.isChecked()):
            self.plt.plot(self.x,self.y,pen=pg.mkPen('#da0037', width=2))
        if(self.ui.check_alcohol_s2.isChecked()):
            self.plt.plot(self.x,self.y5,pen=pg.mkPen('#15dbe6', width=2))
        if(self.ui.check_co_s1.isChecked()):
            self.plt.plot(self.x,self.y1,pen=pg.mkPen('#eb5802', width=2))
        if(self.ui.check_co_s2.isChecked()):
            self.plt.plot(self.x,self.y6,pen=pg.mkPen('#dbf705', width=2))
        if(self.ui.check_dihidrogeno_s1.isChecked()):
            self.plt.plot(self.x,self.y2,pen=pg.mkPen('#04ff00', width=2))
        if(self.ui.check_dihidrogeno_s2.isChecked()):
            self.plt.plot(self.x,self.y7,pen=pg.mkPen('#8f2afa', width=2))
        if(self.ui.check_acetona_s1.isChecked()):
            self.plt.plot(self.x,self.y3,pen=pg.mkPen('#fa2aec', width=2))
        if(self.ui.check_acetona_s2.isChecked()):
            self.plt.plot(self.x,self.y8,pen=pg.mkPen('#fafafa', width=2))
        if(self.ui.check_metano_s1.isChecked()):
            self.plt.plot(self.x,self.y4,pen=pg.mkPen('#32a862', width=2))
        if(self.ui.check_metano_s2.isChecked()):
            self.plt.plot(self.x,self.y9,pen=pg.mkPen('#fc0000', width=2))

    def deshabilitar_clasificar(self):
        self.ui.boton_clasificar.setStyleSheet("image:url(:/images/iconos/Imagen7.png);")
        self.ui.boton_clasificar.setEnabled(False)
    
    def habilitar_clasificar(self):
        self.ui.boton_clasificar.setStyleSheet("QPushButton{"
                                             "image:url(:/images/iconos/Imagen6.png);"
                                             "}"
                                             "QPushButton:hover {"
                                             "image:url(:/images/iconos/Imagen8.png);"
                                             "}")
        self.ui.boton_clasificar.setEnabled(True)
    
    def deshabilitar_entrenar(self):
        self.ui.boton_entrenar.setStyleSheet("image:url(:/images/iconos/Imagen11.png);")
        self.ui.boton_entrenar.setEnabled(False)
    
    def habilitar_entrenar(self):
        self.ui.boton_entrenar.setStyleSheet("QPushButton{"
                                             "image:url(:/images/iconos/Imagen9.png);"
                                             "}"
                                             "QPushButton:hover {"
                                             "image:url(:/images/iconos/Imagen10.png);"
                                             "}")
        self.ui.boton_entrenar.setEnabled(True)

    def apagar_titulo_clasificar(self):
        self.ui.label_14.setStyleSheet("color:rgb(40,40,40);"
                                       "font:87 12pt 'cooper black';")

    def encender_titulo_clasificar(self):
        self.ui.label_14.setStyleSheet("color:rgb(146,208,80);"
                                       "font:87 12pt 'cooper black';")

    def imprimir_categoria(self, categoria):
        self.ui.label_categoria.setText(str(categoria))
        self.ui.label_categoria.setStyleSheet("color:rgb(146,208,80);"
                                              "font:87 20pt 'cooper black';")

    def borrar_categoria(self):
        self.ui.label_categoria.setText("??")
        self.ui.label_categoria.setStyleSheet("color:rgb(17,17,17);"
                                              "font:87 20pt 'cooper black';")

    def deshabilitar_generar_datos(self):
        self.ui.boton_generar_datos.setStyleSheet("background-color:rgb(17,17,17);"
                                          "font:87 12pt 'cooper black';"
                                          "color:rgb(44,44,44);")
        self.ui.boton_generar_datos.setEnabled(False)

    def deshabilitar_borrar_muestra(self):
        self.ui.boton_borrar_muestra.setStyleSheet("background-color:rgb(17,17,17);"
                                          "font:87 12pt 'cooper black';"
                                          "color:rgb(44,44,44);")
        self.ui.boton_borrar_muestra.setEnabled(False)

    def habilitar_generar_datos(self):
        self.ui.boton_generar_datos.setEnabled(True)
        self.ui.boton_generar_datos.setStyleSheet("QPushButton{"
                                             "background-color:rgb(146,208,80);"
                                             "font:87 12pt 'cooper black';"
                                             "color:rgb(17,17,17);"
                                             "}"
                                             "QPushButton:hover {"
                                             "background-color:rgb(17,17,17);"
                                             "font:87 12pt 'cooper black';"
                                             "color:rgb(146,208,80);"
                                             "}")

    def habilitar_borrar_muestra(self):
        self.ui.boton_borrar_muestra.setEnabled(True)
        self.ui.boton_borrar_muestra.setStyleSheet("QPushButton{"
                                             "background-color:rgb(146,208,80);"
                                             "font:87 12pt 'cooper black';"
                                             "color:rgb(17,17,17);"
                                             "}"
                                             "QPushButton:hover {"
                                             "background-color:rgb(17,17,17);"
                                             "font:87 12pt 'cooper black';"
                                             "color:rgb(146,208,80);"
                                             "}")

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
        file_names=os.listdir('dataframe')
        self.serial.waitForReadyRead(100)
        self.port = self.ui.comboBox_puerto.currentText()
        self.baud = self.ui.comboBox_baudrate.currentText()
        self.serial.setBaudRate(int(self.baud))
        self.serial.setPortName(self.port)
        self.serial.open(QIODevice.ReadWrite)
        self.ui.boton_desconectar.show()
        self.ui.boton_desconectar.setEnabled(True)
        self.ui.boton_conectar.setStyleSheet("background-color:rgb(17,17,17);"
                                             "font:87 12pt 'cooper black';"
                                             "color:rgb(218,0,55);")
        self.ui.boton_conectar.setText("CONECTADO")
        self.ui.boton_conectar.setEnabled(False)
        if file_names:
            self.entrenar_red()
        else:
            pass
    
    def entrenar_red(self):
        dt_frame=pd.read_csv('dataframe/dataframe.csv')
        Y=dt_frame.categoria
        X_raw=dt_frame[['prom_alcohol_s1[PPM]','razon_max_value_metano_alcohol_s1','promElevation_alcohol_s1[PPM]']]
        X=pd.DataFrame()
        X[['prom_alcohol_s1[PPM]','razon_max_value_metano_alcohol_s1','promElevation_alcohol_s1[PPM]']]=self.sc.fit_transform(
            X_raw[['prom_alcohol_s1[PPM]','razon_max_value_metano_alcohol_s1','promElevation_alcohol_s1[PPM]']])
        print(X.head(5))
        self.MLP_classifier.fit(X,Y)
        print("red entrenada")

    def serial_disconnect(self):
        self.ui.boton_desconectar.hide()
        self.ui.boton_conectar.setEnabled(True)
        self.ui.boton_conectar.setText("CONECTAR")
        self.ui.boton_conectar.setStyleSheet("QPushButton{"
                                             "background-color:rgb(146,208,80);"
                                             "font:87 12pt 'cooper black';"
                                             "color:rgb(17,17,17);"
                                             "}"
                                             "QPushButton:hover {"
                                             "background-color:rgb(17,17,17);"
                                             "font:87 12pt 'cooper black';"
                                             "color:rgb(146,208,80);"
                                             "}")
        self.ui.boton_desconectar.setEnabled(False)
        self.deshabilitarAjusteAmbiental()
        self.door0=True
        self.serial.close()

    def read_data(self):
        if not self.serial.canReadLine(): return
        rx = self.serial.readLine()
        x=str(rx, 'utf-8').strip()
        x=x.split(',')
        fin=int(x[10])
        #print(f'{x[0]} -- {x[1]} -- {x[2]} -- {x[3]} -- {x[4]} -- {x[5]} -- {x[6]} -- {x[7]} -- {x[8]} -- {x[9]} -- {x[10]}')
        if(fin==0):
            self.y = self.y[1:]
            self.y1 = self.y1[1:]
            self.y2 = self.y2[1:]
            self.y3 = self.y3[1:]
            self.y4 = self.y4[1:]
            self.y5 = self.y5[1:]
            self.y6 = self.y6[1:]
            self.y7 = self.y7[1:]
            self.y8 = self.y8[1:]
            self.y9 = self.y9[1:]
            print(f"offset: {self.offset}")
            self.y.append(float(x[0])-self.offset)
            self.y1.append(float(x[1]))
            self.y2.append(float(x[2]))
            self.y3.append(float(x[3]))
            self.y4.append(float(x[4])-self.offset2)
            self.y5.append(float(x[5]))
            self.y6.append(float(x[6]))
            self.y7.append(float(x[7]))
            self.y8.append(float(x[8]))
            self.y9.append(float(x[9]))
            self.plt.clear()
            if(self.ui.check_alcohol_s1.isChecked()):
                self.plt.plot(self.x,self.y,pen=pg.mkPen('#da0037', width=2))
            if(self.ui.check_alcohol_s2.isChecked()):
                self.plt.plot(self.x,self.y5,pen=pg.mkPen('#15dbe6', width=2))
            if(self.ui.check_co_s1.isChecked()):
                self.plt.plot(self.x,self.y1,pen=pg.mkPen('#eb5802', width=2))
            if(self.ui.check_co_s2.isChecked()):
                self.plt.plot(self.x,self.y6,pen=pg.mkPen('#dbf705', width=2))
            if(self.ui.check_dihidrogeno_s1.isChecked()):
                self.plt.plot(self.x,self.y2,pen=pg.mkPen('#04ff00', width=2))
            if(self.ui.check_dihidrogeno_s2.isChecked()):
                self.plt.plot(self.x,self.y7,pen=pg.mkPen('#8f2afa', width=2))
            if(self.ui.check_acetona_s1.isChecked()):
                self.plt.plot(self.x,self.y3,pen=pg.mkPen('#fa2aec', width=2))
            if(self.ui.check_acetona_s2.isChecked()):
                self.plt.plot(self.x,self.y8,pen=pg.mkPen('#fafafa', width=2))
            if(self.ui.check_metano_s1.isChecked()):
                self.plt.plot(self.x,self.y4,pen=pg.mkPen('#32a862', width=2))
            if(self.ui.check_metano_s2.isChecked()):
                self.plt.plot(self.x,self.y9,pen=pg.mkPen('#fc0000', width=2))
            new_row={
                'ALCOHOL_s1[PPM]':float(x[0])-self.offset,
                'MONOXIDO DE CARBONO_S1[PPM]':float(x[1]),
                'DIHIDROGENO_s1[PPM]':float(x[2]),
                'ACETONA_s1[PPM]':float(x[3]),
                'METANO_s1[PPM]':float(x[4])-self.offset2,
                'ALCOHOL_s2[PPM]':float(x[5]),
                'MONOXIDO DE CARBONO_S2[PPM]':float(x[6]),
                'DIHIDROGENO_s2[PPM]':float(x[7]),
                'ACETONA_s2[PPM]':float(x[8]),
                'METANO_s2[PPM]':float(x[9]),
            }
            self.df=self.df.append(new_row, ignore_index=True)
            #print(self.df['METANO_s1[PPM]']) #cambiando lectura
            self.habilitar_borrar_muestra()
            self.habilitarAjusteAmbiental()
        elif(fin==1):
            file_names=os.listdir("dataframe")
            if file_names:
                self.habilitar_generar_datos()
            else:
                pass
            self.habilitar_clasificar()
            self.encender_titulo_clasificar()
            self.borrar_generar_datos = True
   
    def generar_rawdata(self):
        file_names=os.listdir('datos_recolectados')
        if file_names: #si el directorio esta lleno
            self.rawdata_counter=0
            counter=0
            index_list=[]
            for name in file_names:
                flag0=True
                #print(f'name: {name}')
                while flag0:
                    #print(f'name: rawdata{counter}.csv')
                    if name == f'rawdata{counter}.csv':
                        flag0=False
                        #print(f'counter: {counter}')
                        index_list.append(counter)
                        counter=0
                    else:
                        counter=counter+1
            aux_counter=0
            index_list.sort()
            flag1=True
            while flag1:
                try:
                    if index_list[aux_counter] == aux_counter:
                        aux_counter=aux_counter+1
                    else:
                        flag1=False
                except:
                    flag1=False
            #print(f'index list: {aux_counter}')
            self.rawdata_counter=aux_counter
            aux_counter=0
            index_list=[]
            df_rawdata=self.df.iloc[self.infLimit:self.supLimit,:]
            df_rawdata.to_csv(f'datos_recolectados/rawdata{self.rawdata_counter}.csv')     
        else: #si el directorio esta vacio
            df_rawdata=self.df.iloc[self.infLimit:self.supLimit,:]
            df_rawdata.to_csv('datos_recolectados/rawdata0.csv')
    
    def resetear_rawdata(self):
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

    def calcular_values_dataframe(self, train_invoked=False):
        categoria=self.ui.comboBox_categoria.currentText()
        try:
            tamano=float(self.ui.lineEdit_tamano.text().strip())
            self.generar_rawdata()
            self.generar_dataframe(tamano, categoria)
            if train_invoked==False:
                self.resetear_dataframe()
                self.resetear_rawdata()
                self.limpiar_grafica()
                self.deshabilitar_generar_datos()
                self.deshabilitar_borrar_muestra()
                self.deshabilitar_clasificar()
                self.deshabilitar_entrenar()
                self.apagar_titulo_clasificar()
                self.borrar_categoria()
            else:
                pass
        except:
            mensaje=QMessageBox()
            mensaje.setWindowTitle("Error")
            mensaje.setIcon(QMessageBox.Warning)
            mensaje.setText("ingresa todos los campos")
            mensaje.exec_()

    def generar_dataframe(self, size, cat):
        df_dataframe=self.df.iloc[self.infLimit:self.supLimit,:]
        promedio_alcohol_s1=df_dataframe['ALCOHOL_s1[PPM]'].mean()
        max_alcohol_s1=df_dataframe['ALCOHOL_s1[PPM]'].max()
        promedio_alcohol_s2=df_dataframe['ALCOHOL_s2[PPM]'].mean()
        max_alcohol_s2=df_dataframe['ALCOHOL_s2[PPM]'].max()
        promedio_metano_s1=df_dataframe['METANO_s1[PPM]'].mean()
        max_metano_s1=df_dataframe['METANO_s1[PPM]'].max()
        promedio_metano_s2=df_dataframe['METANO_s2[PPM]'].mean()
        max_metano_s2=df_dataframe['METANO_s2[PPM]'].max()
        razon_max_metano_alcohol_s1=max_metano_s1/max_alcohol_s1
        razon_max_metano_alcohol_s2=max_metano_s2/max_alcohol_s2
        frstMeasure_alcohol_s1=df_dataframe['ALCOHOL_s1[PPM]'].iloc[0]
        frstMeasure_alcohol_s2=df_dataframe['ALCOHOL_s2[PPM]'].iloc[0]
        frstMeasure_metano_s1=df_dataframe['METANO_s1[PPM]'].iloc[0]
        frstMeasure_metano_s2=df_dataframe['METANO_s2[PPM]'].iloc[0]
        crvElevation_alcohol_s1=max_alcohol_s1-frstMeasure_alcohol_s1
        promElevation_alcohol_s1=promedio_alcohol_s1-frstMeasure_alcohol_s1
        crvElevation_alcohol_s2=max_alcohol_s2-frstMeasure_alcohol_s2
        promElevation_alcohol_s2=promedio_alcohol_s2-frstMeasure_alcohol_s2
        crvElevation_metano_s1=max_metano_s1-frstMeasure_metano_s1
        promElevation_metano_s1=promedio_metano_s1-frstMeasure_metano_s1
        crvElevation_metano_s2=max_metano_s2-frstMeasure_metano_s2
        promElevation_metano_s2=promedio_metano_s2-frstMeasure_metano_s2

        #print(f'{promedio_alcohol_s1} -- {max_alcohol_s1} -- {promedio_alcohol_s2} -- {max_alcohol_s2} -- {promedio_metano_s1} -- {max_metano_s1} -- {promedio_metano_s2} -- {max_metano_s2} -- {razon_max_metano_alcohol_s1} -- {razon_max_metano_alcohol_s2} -- {tamano} -- {categoria}')
        file_names2=os.listdir('dataframe')
        new_row2={
            'identifier':self.rawdata_counter,
            'prom_alcohol_s1[PPM]':promedio_alcohol_s1,
            'max_val_alcohol_s1[PPM]':max_alcohol_s1,
            'prom_alcohol_s2[PPM]':promedio_alcohol_s2,
            'max_val_alcohol_s2[PPM]':max_alcohol_s2,
            'prom_metano_s1[PPM]':promedio_metano_s1,
            'max_val_metano_s1[PPM]':max_metano_s1,
            'prom_metano_s2[PPM]':promedio_metano_s2,
            'max_val_metano_s2[PPM]':max_metano_s2,
            'razon_max_value_metano_alcohol_s1':razon_max_metano_alcohol_s1,
            'razon_max_value_metano_alcohol_s2':razon_max_metano_alcohol_s2,
            'tamaño[cm]':size,
            'categoria':cat,
            'frstMeasure_alcohol_s1':frstMeasure_alcohol_s1,
            'frstMeasure_alcohol_s2':frstMeasure_alcohol_s2,
            'frstMeasure_metano_s1':frstMeasure_metano_s1,
            'frstMeasure_metano_s2':frstMeasure_metano_s2,
            'crvElevation_alcohol_s1[PPM]':crvElevation_alcohol_s1,
            'promElevation_alcohol_s1[PPM]':promElevation_alcohol_s1,
            'crvElevation_alcohol_s2[PPM]':crvElevation_alcohol_s2,
            'promElevation_alcohol_s2[PPM]':promElevation_alcohol_s2,
            'crvElevation_metano_s1[PPM]':crvElevation_metano_s1,
            'promElevation_metano_s1[PPM]':promElevation_metano_s1,
            'crvElevation_metano_s2[PPM]':crvElevation_metano_s2,
            'promElevation_metano_s2[PPM]':promElevation_metano_s2     
        }
        #print(self.df2)
        if file_names2:
            df3=pd.read_csv('dataframe/dataframe.csv')
            frame_index=0
            for i in range(0,df3.shape[0]):
                if self.rawdata_counter-1 == df3.iloc[i,0]:
                    frame_index=i+1
                else:
                    pass
            lista_name=os.listdir('datos_recolectados')
            if df3.shape[0]+1 == len(lista_name):
                print("añadiendo a dataframe")       
                df3=df3.append(new_row2, ignore_index=True)
                df3.to_csv('dataframe/dataframe.csv', index=False)
                print(f'frame index: {frame_index}')
            else:
                print(f'frame index: {frame_index}')
                print(f'size: {size}')
                df3.loc[df3['identifier']==frame_index, 'prom_alcohol_s1[PPM]']=promedio_alcohol_s1
                df3.loc[df3['identifier']==frame_index, 'max_val_alcohol_s1[PPM]']=max_alcohol_s1
                df3.loc[df3['identifier']==frame_index, 'prom_alcohol_s2[PPM]']=promedio_alcohol_s2
                df3.loc[df3['identifier']==frame_index, 'max_val_alcohol_s2[PPM]']=max_alcohol_s2
                df3.loc[df3['identifier']==frame_index, 'prom_metano_s1[PPM]']=promedio_metano_s1
                df3.loc[df3['identifier']==frame_index, 'max_val_metano_s1[PPM]']=max_metano_s2
                df3.loc[df3['identifier']==frame_index, 'prom_metano_s2[PPM]']=promedio_metano_s2
                df3.loc[df3['identifier']==frame_index, 'max_val_metano_s2[PPM]']=max_metano_s2
                df3.loc[df3['identifier']==frame_index, 'razon_max_value_metano_alcohol_s1']=razon_max_metano_alcohol_s1
                df3.loc[df3['identifier']==frame_index, 'razon_max_value_metano_alcohol_s2']=razon_max_metano_alcohol_s2
                df3.loc[df3['identifier']==frame_index, 'tamaño[cm]']=size
                df3.loc[df3['identifier']==frame_index, 'categoria']=cat
                df3.loc[df3['identifier']==frame_index, 'frstMeasure_alcohol_s1']=frstMeasure_alcohol_s1
                df3.loc[df3['identifier']==frame_index, 'frstMeasure_alcohol_s2']=frstMeasure_alcohol_s2
                df3.loc[df3['identifier']==frame_index, 'frstMeasure_metano_s1']=frstMeasure_metano_s1
                df3.loc[df3['identifier']==frame_index, 'frstMeasure_metano_s2']=frstMeasure_metano_s2
                df3.loc[df3['identifier']==frame_index, 'crvElevation_alcohol_s1[PPM]']=crvElevation_alcohol_s1
                df3.loc[df3['identifier']==frame_index, 'promElevation_alcohol_s1[PPM]']=promElevation_alcohol_s1
                df3.loc[df3['identifier']==frame_index, 'crvElevation_alcohol_s2[PPM]']=crvElevation_alcohol_s2
                df3.loc[df3['identifier']==frame_index, 'promElevation_alcohol_s2[PPM]']=promElevation_alcohol_s2
                df3.loc[df3['identifier']==frame_index, 'crvElevation_metano_s1[PPM]']=crvElevation_metano_s1
                df3.loc[df3['identifier']==frame_index, 'promElevation_metano_s1[PPM]']=promElevation_metano_s1
                df3.loc[df3['identifier']==frame_index, 'crvElevation_metano_s2[PPM]']=crvElevation_metano_s2
                df3.loc[df3['identifier']==frame_index, 'promElevation_metano_s2[PPM]']=promElevation_metano_s2
                df3.to_csv('dataframe/dataframe.csv', index=False)
        else:
            self.df2=self.df2.append(new_row2, ignore_index=True)
            self.df2.to_csv('dataframe/dataframe.csv', index=False)
    
    def resetear_dataframe(self):
        self.df2 =pd.DataFrame({
            'identifier':[],
            'prom_alcohol_s1[PPM]':[],
            'max_val_alcohol_s1[PPM]':[],
            'prom_alcohol_s2[PPM]':[],
            'max_val_alcohol_s2[PPM]':[],
            'prom_metano_s1[PPM]':[],
            'max_val_metano_s1[PPM]':[],
            'prom_metano_s2[PPM]':[],
            'max_val_metano_s2[PPM]':[],
            'razon_max_value_metano_alcohol_s1':[],
            'razon_max_value_metano_alcohol_s2':[],
            'tamaño[cm]':[],
            'categoria':[],
            'frstMeasure_alcohol_s1':[],
            'frstMeasure_alcohol_s2':[],
            'frstMeasure_metano_s1':[],
            'frstMeasure_metano_s2':[],
            'crvElevation_alcohol_s1':[],
            'promElevation_alcohol_s1':[],
            'crvElevation_alcohol_s2':[],
            'promElevation_alcohol_s2':[],
            'crvElevation_metano_s1':[],
            'promElevation_metano_s1':[],
            'crvElevation_metano_s2':[],
            'promElevation_metano_s2':[]
            })

    def limpiar_grafica(self):
        self.x = list(np.linspace(0,300,300))
        self.y = list(np.linspace(0,0,300))
        self.y1 = list(np.linspace(0,0,300))
        self.y2 = list(np.linspace(0,0,300))
        self.y3 = list(np.linspace(0,0,300))
        self.y4 = list(np.linspace(0,0,300))
        self.y5 = list(np.linspace(0,0,300))
        self.y6 = list(np.linspace(0,0,300))
        self.y7 = list(np.linspace(0,0,300))
        self.y8 = list(np.linspace(0,0,300))
        self.y9 = list(np.linspace(0,0,300))
        self.plt.clear()
        if(self.ui.check_alcohol_s1.isChecked()):
            self.plt.plot(self.x,self.y,pen=pg.mkPen('#da0037', width=2))
        if(self.ui.check_alcohol_s2.isChecked()):
            self.plt.plot(self.x,self.y5,pen=pg.mkPen('#15dbe6', width=2))
        if(self.ui.check_co_s1.isChecked()):
            self.plt.plot(self.x,self.y1,pen=pg.mkPen('#eb5802', width=2))
        if(self.ui.check_co_s2.isChecked()):
            self.plt.plot(self.x,self.y6,pen=pg.mkPen('#dbf705', width=2))
        if(self.ui.check_dihidrogeno_s1.isChecked()):
            self.plt.plot(self.x,self.y2,pen=pg.mkPen('#04ff00', width=2))
        if(self.ui.check_dihidrogeno_s2.isChecked()):
            self.plt.plot(self.x,self.y7,pen=pg.mkPen('#8f2afa', width=2))
        if(self.ui.check_acetona_s1.isChecked()):
            self.plt.plot(self.x,self.y3,pen=pg.mkPen('#fa2aec', width=2))
        if(self.ui.check_acetona_s2.isChecked()):
            self.plt.plot(self.x,self.y8,pen=pg.mkPen('#fafafa', width=2))
        if(self.ui.check_metano_s1.isChecked()):
            self.plt.plot(self.x,self.y4,pen=pg.mkPen('#32a862', width=2))
        if(self.ui.check_metano_s2.isChecked()):
            self.plt.plot(self.x,self.y9,pen=pg.mkPen('#fc0000', width=2))

    def borrar_muestra(self):
        if self.borrar_generar_datos:
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
                'METANO_s2[PPM]':[]
            })
            self.deshabilitar_borrar_muestra()
            self.deshabilitar_generar_datos()
            self.deshabilitar_clasificar()
            self.deshabilitar_entrenar()
            self.apagar_titulo_clasificar()
            self.borrar_categoria()
            self.borrar_generar_datos = False
            self.limpiar_grafica()
            self.door1=True
            
        else:
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
                'METANO_s2[PPM]':[]
            })
            self.deshabilitar_borrar_muestra()
            self.deshabilitar_clasificar()
            self.deshabilitar_entrenar()
            self.apagar_titulo_clasificar()
            self.borrar_categoria()
            self.limpiar_grafica()

    def control_normalizar(self):
        self.showNormal()
        self.ui.boton_normalizar.hide()
        self.ui.boton_maximizar.show()
    
    def clasificar(self):
        file_names=os.listdir('dataframe')
        if file_names:
            if self.door1:
                self.habilitar_entrenar()
                self.door1=False
            df_clasificar=self.df.iloc[self.infLimit:self.supLimit,:]
            #print(df_clasificar.shape)
            promedio_alcohol_s1=df_clasificar['ALCOHOL_s1[PPM]'].mean()
            max_alcohol_s1=df_clasificar['ALCOHOL_s1[PPM]'].max()
            max_metano_s1=df_clasificar['METANO_s1[PPM]'].max()
            razon_max_metano_alcohol_s1=max_metano_s1/max_alcohol_s1
            frstMeasure_alcohol_s1=df_clasificar['ALCOHOL_s1[PPM]'].iloc[0]
            promElevation_alcohol_s1=promedio_alcohol_s1-frstMeasure_alcohol_s1
            X_test=pd.DataFrame({
                'prom_alcohol_s1[PPM]':[promedio_alcohol_s1],
                'razon_max_value_metano_alcohol_s1':[razon_max_metano_alcohol_s1],
                'promElevation_alcohol_s1[PPM]':[promElevation_alcohol_s1]
            })
            X_test[['prom_alcohol_s1[PPM]',
                    'razon_max_value_metano_alcohol_s1',
                    'promElevation_alcohol_s1[PPM]']]=self.sc.transform(X_test[['prom_alcohol_s1[PPM]',
                                                                                'razon_max_value_metano_alcohol_s1',
                                                                                'promElevation_alcohol_s1[PPM]']])
            print(X_test)
            Y_pred=self.MLP_classifier.predict(X_test)
            self.imprimir_categoria(Y_pred[0])
        else:
            if self.door1:
                self.habilitar_entrenar()
                self.door1=False
            self.imprimir_categoria("??")
            self.habilitar_entrenar()

    def entrenar(self):
        file_names=os.listdir('dataframe')
        if file_names:
            pass
        else:
            self.calcular_values_dataframe(True)
        dt_frame=pd.read_csv('dataframe/dataframe.csv')
        X_raw=dt_frame[['prom_alcohol_s1[PPM]','razon_max_value_metano_alcohol_s1','promElevation_alcohol_s1[PPM]','categoria']]
        df_entrenar=self.df.iloc[self.infLimit:self.supLimit,:]
        promedio_alcohol_s1=df_entrenar['ALCOHOL_s1[PPM]'].mean()
        max_alcohol_s1=df_entrenar['ALCOHOL_s1[PPM]'].max()
        max_metano_s1=df_entrenar['METANO_s1[PPM]'].max()
        razon_max_metano_alcohol_s1=max_metano_s1/max_alcohol_s1
        frstMeasure_alcohol_s1=df_entrenar['ALCOHOL_s1[PPM]'].iloc[0]
        promElevation_alcohol_s1=promedio_alcohol_s1-frstMeasure_alcohol_s1
        categoria=self.ui.comboBox_categoria.currentText()
        X_last={
            'prom_alcohol_s1[PPM]':promedio_alcohol_s1,
            'razon_max_value_metano_alcohol_s1':razon_max_metano_alcohol_s1,
            'promElevation_alcohol_s1[PPM]':promElevation_alcohol_s1,
            'categoria':int(categoria)
        }
        X_raw=X_raw.append(X_last, ignore_index=True)
        Y=X_raw.categoria
        X_raw.drop(['categoria'],axis=1)
        X=pd.DataFrame()
        X[['prom_alcohol_s1[PPM]','razon_max_value_metano_alcohol_s1','promElevation_alcohol_s1[PPM]']]=self.sc.fit_transform(
            X_raw[['prom_alcohol_s1[PPM]','razon_max_value_metano_alcohol_s1','promElevation_alcohol_s1[PPM]']])
        print(X.tail(5))
        self.MLP_classifier.fit(X,Y)
        self.ui.label_categoria.setText("red entrenada")
        self.ui.label_categoria.setStyleSheet("color:rgb(218,0,55);"
                                              "font:87 20pt 'cooper black';")
        self.deshabilitar_entrenar()

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