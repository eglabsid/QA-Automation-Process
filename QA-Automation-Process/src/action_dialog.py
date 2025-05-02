
import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QCursor, QIntValidator
from PyQt5.QtCore import QTimer, Qt


dialog_ui = 'src/action_dialog.ui'

class ActionDialog(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        
        uic.loadUi(dialog_ui, self)
        
        self.setLayout(self.main_layout)
        
        self.setWindowTitle("동작 추가")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.resize(400 , 200)
        
        self.mousePos = [0,0]
        
        self.input_toggle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.track_mouse)
        self.timer.start(100)
        
        self.confirm_btn.clicked.connect(self.accept_btn)
        self.cancel_btn.clicked.connect(self.reject)
        self.radio_pos.toggled.connect(self.update_radio_btn)
        self.radio_key.toggled.connect(self.update_radio_btn)
        self.radio_pos.setChecked(True)
        
        self.input_key.setMaxLength(15)
        self.input_key.setPlaceholderText("특정 Key")
        
        # 마우스 리스너 시작
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()
        
    def update_radio_btn(self):
        if self.radio_pos.isChecked():
            self.input_toggle = 0
            self.input_key.setDisabled(True)
            
        if self.radio_key.isChecked():
            self.input_toggle = 1
            self.input_key.setDisabled(False)

    def track_mouse(self):
        mouse_pos = QCursor.pos()
        self.current_pos.setText(f"현재 마우스 좌표 : ({mouse_pos.x()}, {mouse_pos.y()})")
        if self.radio_pos.isChecked():
            self.mouse_pos.setText(f"( {mouse_pos.x()}, {mouse_pos.y()} )")
            self.mousePos = [mouse_pos.x(),mouse_pos.y()]
        else:
            self.mouse_pos.setText(f"( x, y )")
            self.mousePos = [0,0]

    def on_click(self, x, y, button, pressed):
        if pressed and button == mouse.Button.right :
            if self.input_toggle == 0:
                self.accept()

    def closeEvent(self, event):
        # 애플리케이션 종료 시 리스너 중지
        self.listener.stop()
        event.accept()
            
    def accept_btn(self):
        if self.input_toggle == 0:
            if not self.mousePos[0] or not self.mousePos[1]:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText("좌표값이 비어있습니다.")
                msg.setWindowTitle("Error")
                msg.resize(600,200)
                msg.exec_()
            else:
                self.accept()

        if self.input_toggle == 1:
            if not self.input_key.text():
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText("키 값이 비어있습니다.")
                msg.setWindowTitle("Error")
                msg.resize(600, 200)
                msg.exec_()
            else:
                self.accept()
                
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = ActionDialog()
    dialog.show()
    sys.exit(app.exec_())        