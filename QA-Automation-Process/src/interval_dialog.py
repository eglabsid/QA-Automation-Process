
import sys, os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QDoubleValidator

dialog_ui = 'src/interval_dialog.ui'

class IntervalDialog(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        
        uic.loadUi(dialog_ui, self)
        
        self.setLayout(self.main_layout)
        
        self.setWindowTitle("행동간격시간 설정")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.resize(400, 200)
        
        self.interval_line.setValidator(QDoubleValidator(0.0, 10000.0, 3, self))
        self.interval_line.setMaxLength(5)
        
        self.confirm_btn.clicked.connect(self.accept_btn)
        self.confirm_btn.setDefault(True)
        self.cancel_btn.clicked.connect(self.reject)


    def accept_btn(self):
        if not self.interval_line:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("입력값이 비어있습니다.")
            msg.setWindowTitle("Error")
            msg.resize(600, 200)
            msg.exec_()
        else:
            self.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = IntervalDialog()
    dialog.show()
    sys.exit(app.exec_())        