
import sys, os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon

dialog_ui = 'src/image_dialog.ui'

class ImageDialog(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self._imgPath = None
        
        uic.loadUi(dialog_ui, self)
        
        self.setLayout(self.main_layout)
        
        self.setWindowTitle("이미지 클릭 추가")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.resize(400, 200)
        
        self.img_path.setReadOnly(True)
        self.img_path.setPlaceholderText("이미지 경로")

        self.browse_btn.clicked.connect(self.select_img)
        
        self.confidence.setRange(50,100)
        self.confidence.setValue(100)

        self.confirm_btn.clicked.connect(self.accept_btn)
        self.confirm_btn.setDefault(True)
        
        self.cancel_btn.clicked.connect(self.reject)
        

    def select_img(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly  # 파일을 읽기 전용으로 열기
        file_filter = "Images (*.png *.jpg *.jpeg *.bmp)"  # 이미지 파일 필터 설정
        imgPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", file_filter, options=options)

        if imgPath:
            imgPath = imgPath.split('.')[0]
            self.img_path.setText(imgPath)
            self._imgPath = imgPath
        else:
            return None
    
    
    def accept_btn(self):
        if not self._imgPath:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("선택한 이미지가 없습니다.")
            msg.setWindowTitle("Error")
            msg.resize(600, 200)
            msg.exec_()
        else:
            self.accept()
    
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = ImageDialog()
    dialog.show()
    sys.exit(app.exec_())        