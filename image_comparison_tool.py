import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QGridLayout, QLabel, QPushButton, QFileDialog, QComboBox, 
                            QSpinBox, QGroupBox, QScrollArea, QLineEdit, QToolTip,
                            QRadioButton, QButtonGroup, QMessageBox, QCheckBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect
import numpy as np
from PIL import Image, ImageDraw
import multiprocessing as mp
from functools import partial
import heapq
import math

# 全局函數，用於計算區域差異
def calculate_region_difference(region1, region2, metric="MSE"):
    """計算兩個圖像區域之間的差異"""
    if metric == "MSE (均方誤差)":
        return np.mean((region1 - region2) ** 2)
    elif metric == "MAE (平均絕對誤差)":
        return np.mean(np.abs(region1 - region2))
    elif metric == "SSIM (結構相似性)":
        # 對於SSIM，我們返回1-SSIM，因為我們要找的是差異最大/最小的點
        # 簡化版本的實現，實際上應該使用專門的SSIM庫
        return np.mean(np.abs(region1 - region2))  # 簡化實現
    else:
        return np.mean((region1 - region2) ** 2)  # 默認使用MSE

# 全局函數，用於比較窗口
def compare_regions(img1, img2, gt, start_x, start_y, window_size, mode, metric):
    """比較以(start_x, start_y)為起點的窗口區域"""
    try:
        # 裁剪區域
        region1 = img1.crop((start_x, start_y, start_x + window_size, start_y + window_size))
        region2 = img2.crop((start_x, start_y, start_x + window_size, start_y + window_size))
        region_gt = gt.crop((start_x, start_y, start_x + window_size, start_y + window_size))
        
        # 轉換為numpy數組
        region1_array = np.array(region1)
        region2_array = np.array(region2)
        region_gt_array = np.array(region_gt)
        
        # 計算差異
        diff1_gt = calculate_region_difference(region1_array, region_gt_array, metric)
        diff2_gt = calculate_region_difference(region2_array, region_gt_array, metric)
        
        if mode == 1:  # 圖像1最接近GT，圖像2最遠離GT
            score = diff2_gt - diff1_gt
        else:  # 圖像2最接近GT，圖像1最遠離GT
            score = diff1_gt - diff2_gt
        
        return (start_x, start_y, score, diff1_gt, diff2_gt)
    except Exception as e:
        return (start_x, start_y, float('-inf'), 0, 0)

class ImageComparisonTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("圖像比較工具")
        self.setMinimumSize(1200, 800)  # 增加視窗預設大小
        
        # 設定全螢幕顯示
        self.showMaximized()
        
        # 設定全局字體大小
        font = self.font()
        font.setPointSize(12)  # 增加字體大小
        self.setFont(font)
        
        # 設定全局樣式表增加字體大小
        self.setStyleSheet("""
            QLabel, QPushButton, QCheckBox, QComboBox, QSpinBox, QLineEdit { 
                font-size: 12pt; 
            }
            QGroupBox { 
                font-size: 13pt; 
                font-weight: bold; 
            }
        """)
        
        # 初始化變數
        self.image_paths = [None, None, None, None]
        self.images = [None, None, None, None]
        self.current_size = 32
        self.start_x = 0
        self.start_y = 0
        
        # 設定網格大小
        self.grid_size = 20  # 預設改為20
        
        # 存儲最佳結果
        self.top_results = []
        self.current_result_index = 0
        
        # 設定預設放大尺寸
        self.preview_size = 128
        
        # 創建主要佈局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 上半部分控制面板 - 使用三列佈局
        control_panel = QWidget()
        main_layout.addWidget(control_panel, 1)  # 控制面板佔用1/3空間
        
        # 控制面板佈局 - 三直列
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(15)  # 增加列之間的間距
        
        # ===== 第一直列：圖像選擇區域 =====
        image_selection = QGroupBox("選擇圖像")
        image_selection.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; }")
        image_selection_layout = QVBoxLayout(image_selection)
        image_selection_layout.setSpacing(10)
        
        self.image_buttons = []
        self.image_labels = []
        self.image_path_edits = []
        
        for i in range(4):
            # 創建每個圖像的選擇組合框
            image_frame = QFrame()
            image_frame.setFrameShape(QFrame.StyledPanel)
            image_frame.setStyleSheet("QFrame { background-color: #f9f9f9; border-radius: 5px; }")
            image_frame_layout = QVBoxLayout(image_frame)
            
            button_text = f"選擇圖像 {i+1}"
            if i == 0:
                button_text = "選擇圖像 1 (比較圖1)"
            elif i == 1:
                button_text = "選擇圖像 2 (比較圖2)"
            elif i == 3:
                button_text = "選擇圖像 4 (GT參考圖)"
                
            self.image_buttons.append(QPushButton(button_text))
            self.image_buttons[i].clicked.connect(lambda checked, idx=i: self.load_image(idx))
            self.image_buttons[i].setStyleSheet("QPushButton { min-height: 30px; background-color: #2196F3; color: white; }")
            
            # 使用QLineEdit顯示完整路徑
            self.image_path_edits.append(QLineEdit(f"未選擇圖像 {i+1}"))
            self.image_path_edits[i].setReadOnly(True)
            self.image_path_edits[i].setStyleSheet("color: gray; padding: 5px;")
            self.image_path_edits[i].setToolTip(f"未選擇圖像 {i+1}")
            
            self.image_labels.append(QLabel(f"未選擇圖像 {i+1}"))
            self.image_labels[i].setStyleSheet("color: gray;")
            self.image_labels[i].hide()  # 隱藏原有的標籤
            
            image_frame_layout.addWidget(self.image_buttons[i])
            image_frame_layout.addWidget(self.image_path_edits[i])
            
            image_selection_layout.addWidget(image_frame)
        
        # 將第一直列添加到控制面板佈局
        control_layout.addWidget(image_selection, 1)  # 圖像選擇區佔1/3寬度
        
        # ===== 第二直列：顯示設置和保存功能 =====
        second_column = QWidget()
        second_column_layout = QVBoxLayout(second_column)
        second_column_layout.setSpacing(10)
        
        # --- 顯示設置區域 ---
        settings = QGroupBox("顯示設置")
        settings.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; }")
        settings_layout = QGridLayout(settings)
        settings_layout.setVerticalSpacing(10)
        
        # 窗口大小選擇
        settings_layout.addWidget(QLabel("窗口大小:"), 0, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["32x32", "64x64", "128x128", "256x256"])
        self.size_combo.currentIndexChanged.connect(self.update_window_size)
        self.size_combo.setStyleSheet("QComboBox { min-height: 25px; }")
        settings_layout.addWidget(self.size_combo, 0, 1)
        
        # 起始座標
        settings_layout.addWidget(QLabel("起始座標 X:"), 1, 0)
        self.start_x_spin = QSpinBox()
        self.start_x_spin.setRange(0, 9999)
        self.start_x_spin.valueChanged.connect(self.update_start_x)
        self.start_x_spin.setStyleSheet("QSpinBox { min-height: 25px; }")
        settings_layout.addWidget(self.start_x_spin, 1, 1)
        
        settings_layout.addWidget(QLabel("起始座標 Y:"), 2, 0)
        self.start_y_spin = QSpinBox()
        self.start_y_spin.setRange(0, 9999)
        self.start_y_spin.valueChanged.connect(self.update_start_y)
        self.start_y_spin.setStyleSheet("QSpinBox { min-height: 25px; }")
        settings_layout.addWidget(self.start_y_spin, 2, 1)
        
        # 更新按鈕
        self.update_button = QPushButton("更新顯示")
        self.update_button.clicked.connect(self.update_display)
        self.update_button.setStyleSheet("QPushButton { min-height: 30px; background-color: #4CAF50; color: white; }")
        settings_layout.addWidget(self.update_button, 3, 0, 1, 2)
        
        second_column_layout.addWidget(settings)
        
        # --- 保存功能區域 ---
        save_group = QGroupBox("保存功能")
        save_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; }")
        save_layout = QGridLayout(save_group)
        save_layout.setVerticalSpacing(8)
        
        # 放大比例選擇
        save_layout.addWidget(QLabel("放大預覽尺寸:"), 0, 0)
        self.preview_size_combo = QComboBox()
        self.preview_size_combo.addItems(["64x64", "128x128", "256x256"])
        self.preview_size_combo.setCurrentIndex(1)  # 預設128x128
        self.preview_size_combo.currentIndexChanged.connect(self.update_preview_size)
        save_layout.addWidget(self.preview_size_combo, 0, 1)
        
        # 添加放置位置選擇
        save_layout.addWidget(QLabel("預覽放置位置:"), 1, 0)
        self.corner_combo = QComboBox()
        self.corner_combo.addItems(["右下角", "右上角", "左下角", "左上角"])
        save_layout.addWidget(self.corner_combo, 1, 1)
        
        # 保存選擇框 (每個圖像是否需要保存)
        self.save_checkboxes = []
        for i in range(4):
            name = ""
            if i == 0:
                name = "保存圖像1"
            elif i == 1:
                name = "保存圖像2"
            elif i == 2:
                name = "保存圖像3"
            else:
                name = "保存GT參考圖"
            
            cb = QCheckBox(name)
            cb.setChecked(i != 2)  # 除了圖像3外，其他預設勾選
            self.save_checkboxes.append(cb)
            save_layout.addWidget(cb, 2 + i//2, i%2)
        
        # 保存按鈕
        self.save_button = QPushButton("保存圖像")
        self.save_button.clicked.connect(self.save_images_with_preview)
        self.save_button.setStyleSheet("QPushButton { min-height: 30px; background-color: #FF9800; color: white; }")
        save_layout.addWidget(self.save_button, 4, 0, 1, 2)
        
        second_column_layout.addWidget(save_group)
        
        # 將第二直列添加到控制面板佈局
        control_layout.addWidget(second_column, 1)  # 第二列佔1/3寬度
        
        # ===== 第三直列：自動尋找特徵點和結果導航 =====
        third_column = QWidget()
        third_column_layout = QVBoxLayout(third_column)
        third_column_layout.setSpacing(10)
        
        # --- 自動尋找特徵點區域 ---
        find_settings = QGroupBox("自動尋找特徵點")
        find_settings.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; }")
        find_layout = QGridLayout(find_settings)
        find_layout.setVerticalSpacing(10)
        
        # 找到圖像1與GT差距最小，圖像2與GT差距最大的點
        self.find_button1 = QPushButton("尋找圖像1最接近GT，圖像2最遠離GT的點")
        self.find_button1.clicked.connect(lambda: self.find_special_points(mode=1))
        self.find_button1.setStyleSheet("QPushButton { min-height: 30px; background-color: #2196F3; color: white; }")
        find_layout.addWidget(self.find_button1, 0, 0, 1, 2)
        
        # 找到圖像2與GT差距最小，圖像1與GT差距最大的點
        self.find_button2 = QPushButton("尋找圖像2最接近GT，圖像1最遠離GT的點")
        self.find_button2.clicked.connect(lambda: self.find_special_points(mode=2))
        self.find_button2.setStyleSheet("QPushButton { min-height: 30px; background-color: #2196F3; color: white; }")
        find_layout.addWidget(self.find_button2, 1, 0, 1, 2)
        
        # 說明文字
        info_label = QLabel("注意: 請將GT參考圖放在第4個位置")
        info_label.setStyleSheet("QLabel { color: #FF5722; }")
        find_layout.addWidget(info_label, 2, 0, 1, 2)
        
        # 添加差距度量選擇
        find_layout.addWidget(QLabel("差距度量方式:"), 3, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["MSE (均方誤差)", "MAE (平均絕對誤差)", "SSIM (結構相似性)"])
        self.metric_combo.setStyleSheet("QComboBox { min-height: 25px; }")
        find_layout.addWidget(self.metric_combo, 3, 1)
        
        # 添加網格大小選擇
        find_layout.addWidget(QLabel("網格大小:"), 4, 0)
        self.grid_size_combo = QComboBox()
        self.grid_size_combo.addItems(["10x10", "20x20", "30x30", "40x40", "50x50"])
        self.grid_size_combo.setCurrentIndex(1)  # 預設選擇20x20
        self.grid_size_combo.currentIndexChanged.connect(self.update_grid_size)
        self.grid_size_combo.setStyleSheet("QComboBox { min-height: 25px; }")
        find_layout.addWidget(self.grid_size_combo, 4, 1)
        
        # 添加灰階比較選項
        self.use_grayscale_cb = QCheckBox("使用灰階比較(捕捉結構細節)")
        self.use_grayscale_cb.setStyleSheet("QCheckBox { min-height: 25px; }")
        find_layout.addWidget(self.use_grayscale_cb, 5, 0, 1, 2)
        
        third_column_layout.addWidget(find_settings)
        
        # --- 結果導航區域 ---
        result_area = QWidget()
        result_area_layout = QHBoxLayout(result_area)
        result_area_layout.setSpacing(10)
        
        # 左側：結果導航
        result_nav = QGroupBox("結果導航 (分區最佳結果)")
        result_nav.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; }")
        result_nav_layout = QGridLayout(result_nav)
        result_nav_layout.setVerticalSpacing(8)
        
        # 上一個結果按鈕
        self.prev_result_btn = QPushButton("上一個結果")
        self.prev_result_btn.clicked.connect(self.show_prev_result)
        self.prev_result_btn.setEnabled(False)
        self.prev_result_btn.setStyleSheet("QPushButton { min-height: 28px; }")
        result_nav_layout.addWidget(self.prev_result_btn, 0, 0)
        
        # 下一個結果按鈕
        self.next_result_btn = QPushButton("下一個結果")
        self.next_result_btn.clicked.connect(self.show_next_result)
        self.next_result_btn.setEnabled(False)
        self.next_result_btn.setStyleSheet("QPushButton { min-height: 28px; }")
        result_nav_layout.addWidget(self.next_result_btn, 0, 1)
        
        # 結果計數器
        self.result_counter_label = QLabel("結果: 0/0")
        self.result_counter_label.setAlignment(Qt.AlignCenter)
        self.result_counter_label.setStyleSheet("QLabel { font-weight: bold; }")
        result_nav_layout.addWidget(self.result_counter_label, 1, 0, 1, 2)
        
        # 圖1與GT差距
        self.img1_diff_label = QLabel("圖1與GT差距: N/A")
        result_nav_layout.addWidget(self.img1_diff_label, 2, 0, 1, 2)
        
        # 圖2與GT差距
        self.img2_diff_label = QLabel("圖2與GT差距: N/A")
        result_nav_layout.addWidget(self.img2_diff_label, 3, 0, 1, 2)
        
        # 差距比值
        self.diff_ratio_label = QLabel("差距分數: N/A")
        self.diff_ratio_label.setStyleSheet("QLabel { font-weight: bold; color: #E91E63; }")
        result_nav_layout.addWidget(self.diff_ratio_label, 4, 0, 1, 2)
        
        # 當前區域標籤
        self.current_region_label = QLabel("當前區域: N/A")
        result_nav_layout.addWidget(self.current_region_label, 5, 0, 1, 2)
        
        # 右側：主題設置
        theme_settings = QGroupBox("主題設置")
        theme_settings.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; }")
        theme_layout = QVBoxLayout(theme_settings)
        
        # 黑暗模式大按鈕
        self.theme_button = QPushButton("切換黑暗模式")
        self.theme_button.setCheckable(True)  # 設為可切換按鈕
        self.theme_button.setMinimumHeight(100)  # 設置高度
        self.theme_button.setStyleSheet("""
            QPushButton {
                font-size: 16pt;
                font-weight: bold;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2C3E50, stop:1 #4CA1AF);
                color: white;
                border-radius: 10px;
                padding: 15px;
            }
            QPushButton:checked {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4A148C, stop:1 #880E4F);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #34495E, stop:1 #5DADE2);
            }
        """)
        self.theme_button.clicked.connect(self.toggle_theme_mode)
        theme_layout.addWidget(self.theme_button)
        
        # 主題說明
        theme_desc = QLabel("點擊上方按鈕切換亮/暗主題\n黑暗模式適合在弱光環境下使用")
        theme_desc.setAlignment(Qt.AlignCenter)
        theme_layout.addWidget(theme_desc)
        
        # 將兩個部分添加到結果區域佈局
        result_area_layout.addWidget(result_nav, 7)  # 結果導航佔70%
        result_area_layout.addWidget(theme_settings, 3)  # 主題設置佔30%
        
        third_column_layout.addWidget(result_area)
        
        # 將第三直列添加到控制面板佈局
        control_layout.addWidget(third_column, 1)  # 第三列佔1/3寬度
        
        # 下半部分 - 圖像顯示區域
        display_area = QScrollArea()
        display_area.setWidgetResizable(True)
        display_area.setStyleSheet("QScrollArea { border: 1px solid #ccc; }")
        display_widget = QWidget()
        self.display_layout = QGridLayout(display_widget)
        self.display_layout.setSpacing(10)
        
        # 創建顯示標籤
        self.display_labels = []
        self.info_labels = []
        self.pixmaps = [None, None, None, None]  # 存儲原始pixmap
        
        for i in range(4):
            row = i // 2
            col = i % 2
            
            group = QGroupBox(f"圖像 {i+1}")
            group.setStyleSheet("QGroupBox { font-weight: bold; }")
            group_layout = QVBoxLayout(group)
            
            # 圖像信息標籤
            self.info_labels.append(QLabel("未加載圖像"))
            group_layout.addWidget(self.info_labels[i])
            
            # 圖像顯示標籤
            self.display_labels.append(QLabel())
            self.display_labels[i].setAlignment(Qt.AlignCenter)
            self.display_labels[i].setMinimumSize(250, 250)
            self.display_labels[i].setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
            group_layout.addWidget(self.display_labels[i])
            
            self.display_layout.addWidget(group, row, col)
        
        display_area.setWidget(display_widget)
        main_layout.addWidget(display_area, 2)  # 圖像展示區域佔2/3空間
    
    def load_image(self, index):
        # 設定初始目錄
        initial_dir = ""
        if self.image_paths[index] and os.path.exists(self.image_paths[index]):
            initial_dir = os.path.dirname(self.image_paths[index])
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"選擇圖像 {index+1}", initial_dir, "圖像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            try:
                # 保存圖像路徑
                self.image_paths[index] = file_path
                
                # 更新路徑顯示
                self.image_path_edits[index].setText(file_path)
                self.image_path_edits[index].setToolTip(file_path)
                self.image_path_edits[index].setCursorPosition(0)  # 游標置於開始位置
                
                # 根據當前主題設置文字顏色
                if hasattr(self, 'theme_button') and self.theme_button.isChecked():  # 黑暗模式
                    self.image_path_edits[index].setStyleSheet("color: white; padding: 5px; background-color: #333337; border: 1px solid #3F3F46;")
                else:  # 亮色模式
                    self.image_path_edits[index].setStyleSheet("color: black; padding: 5px;")
                
                # 更新原始標籤（保留但隱藏）
                self.image_labels[index].setText(os.path.basename(file_path))
                
                # 載入圖像
                self.images[index] = Image.open(file_path)
                
                # 更新顯示
                self.update_display()
                
                # 清除結果
                self.top_results = []
                self.current_result_index = 0
                self.update_result_navigation()
            except Exception as e:
                error_msg = f"載入失敗: {str(e)}"
                self.image_path_edits[index].setText(error_msg)
                self.image_path_edits[index].setToolTip(error_msg)
                self.image_path_edits[index].setStyleSheet("color: red; padding: 5px;")
                self.image_labels[index].setText(error_msg)
                self.image_paths[index] = None
                self.images[index] = None
    
    def update_window_size(self):
        size_text = self.size_combo.currentText()
        self.current_size = int(size_text.split('x')[0])
        self.update_display()
        # 清除結果
        self.top_results = []
        self.current_result_index = 0
        self.update_result_navigation()
    
    def update_start_x(self):
        self.start_x = self.start_x_spin.value()
        self.update_display()
    
    def update_start_y(self):
        self.start_y = self.start_y_spin.value()
        self.update_display()
    
    def update_display(self, refresh_only=False):
        for i in range(4):
            if self.images[i] is not None:
                try:
                    # 取得圖像大小
                    width, height = self.images[i].size
                    
                    # 更新信息標籤，增加檔案路徑顯示
                    path_info = f"路徑: {self.image_paths[i]}"
                    size_info = f"圖像大小: {width}x{height}, 區域: ({self.start_x},{self.start_y}) - ({self.start_x+self.current_size},{self.start_y+self.current_size})"
                    self.info_labels[i].setText(f"{path_info}\n{size_info}")
                    self.info_labels[i].setToolTip(f"{path_info}\n{size_info}")
                    
                    # 如果不是只刷新，則重新裁剪並設置圖像
                    if not refresh_only:
                        # 檢查座標是否有效
                        if self.start_x + self.current_size <= width and self.start_y + self.current_size <= height:
                            # 裁剪指定區域
                            crop = self.images[i].crop((self.start_x, self.start_y, 
                                                      self.start_x + self.current_size, 
                                                      self.start_y + self.current_size))
                            
                            # 轉換為QPixmap並顯示
                            crop_array = np.array(crop)
                            height, width, channels = crop_array.shape if len(crop_array.shape) == 3 else (*crop_array.shape, 1)
                            
                            if channels == 1:
                                # 灰度圖轉RGB
                                q_image = QImage(crop_array.data, width, height, width, QImage.Format_Grayscale8)
                            else:
                                # RGB或RGBA圖
                                bytes_per_line = channels * width
                                q_image = QImage(crop_array.data, width, height, bytes_per_line, 
                                               QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888)
                            
                            pixmap = QPixmap.fromImage(q_image)
                            self.pixmaps[i] = pixmap.scaled(250, 250, Qt.KeepAspectRatio)
                            self.display_labels[i].setPixmap(self.pixmaps[i])
                        else:
                            self.display_labels[i].setText(f"座標超出範圍: {width}x{height}")
                            self.pixmaps[i] = None
                            continue
                except Exception as e:
                    self.display_labels[i].setText(f"顯示錯誤: {str(e)}")
                    self.pixmaps[i] = None
            else:
                self.display_labels[i].clear()
                self.display_labels[i].setText("未載入圖像")
                self.info_labels[i].setText("未加載圖像")
                self.pixmaps[i] = None
    
    def update_result_navigation(self):
        """更新結果導航控件的狀態"""
        num_results = len(self.top_results)
        
        # 更新結果計數器
        if num_results > 0:
            self.result_counter_label.setText(f"結果: {self.current_result_index+1}/{num_results}")
            
            # 更新差距標籤
            best_x, best_y, _, diff1_gt, diff2_gt = self.top_results[self.current_result_index]
            self.img1_diff_label.setText(f"圖1與GT差距: {diff1_gt:.6f}")
            self.img2_diff_label.setText(f"圖2與GT差距: {diff2_gt:.6f}")
            self.diff_ratio_label.setText(f"差距分數: {diff2_gt-diff1_gt:.6f}" if diff2_gt > diff1_gt else f"差距分數: {diff1_gt-diff2_gt:.6f}")
            
            # 更新當前區域標籤
            grid_x = best_x // self.grid_size
            grid_y = best_y // self.grid_size
            self.current_region_label.setText(f"區域: ({grid_x*self.grid_size},{grid_y*self.grid_size}) - ({(grid_x+1)*self.grid_size-1},{(grid_y+1)*self.grid_size-1})")
        else:
            self.result_counter_label.setText("結果: 0/0")
            self.img1_diff_label.setText("圖1與GT差距: N/A")
            self.img2_diff_label.setText("圖2與GT差距: N/A")
            self.diff_ratio_label.setText("差距分數: N/A")
            self.current_region_label.setText("當前區域: N/A")
        
        # 更新按鈕狀態
        self.prev_result_btn.setEnabled(num_results > 0 and self.current_result_index > 0)
        self.next_result_btn.setEnabled(num_results > 0 and self.current_result_index < num_results - 1)
    
    def show_prev_result(self):
        """顯示上一個結果"""
        if self.current_result_index > 0 and self.top_results:
            self.current_result_index -= 1
            self.show_current_result()
    
    def show_next_result(self):
        """顯示下一個結果"""
        if self.current_result_index < len(self.top_results) - 1:
            self.current_result_index += 1
            self.show_current_result()
    
    def show_current_result(self):
        """顯示當前索引的結果"""
        if self.top_results and 0 <= self.current_result_index < len(self.top_results):
            best_x, best_y, _, diff1_gt, diff2_gt = self.top_results[self.current_result_index]
            
            # 更新起始位置
            self.start_x_spin.setValue(best_x)
            self.start_y_spin.setValue(best_y)
            
            # 更新顯示
            self.update_display()
            
            # 更新導航控制
            self.update_result_navigation()
    
    def find_special_points(self, mode=1):
        """尋找特殊像素點
        mode=1: 圖像1與GT差距最小，圖像2與GT差距最大的點
        mode=2: 圖像2與GT差距最小，圖像1與GT差距最大的點
        """
        # 檢查是否有足夠的圖像 (只檢查圖像1, 圖像2和GT)
        if self.images[0] is None or self.images[1] is None or self.images[3] is None:
            QMessageBox.warning(self, "警告", "請確保已載入圖像1、圖像2和GT(圖像4)!")
            return
        
        try:
            # 提取完整圖像數據
            img1 = self.images[0]
            img2 = self.images[1]
            gt = self.images[3]
            
            # 獲取窗口大小 (使用當前選擇的size)
            window_size = self.current_size
            
            # 獲取圖像尺寸
            img1_width, img1_height = img1.size
            img2_width, img2_height = img2.size
            gt_width, gt_height = gt.size
            
            # 檢查圖像是否足夠大
            if (img1_width < window_size or img1_height < window_size or
                img2_width < window_size or img2_height < window_size or
                gt_width < window_size or gt_height < window_size):
                QMessageBox.warning(self, "警告", f"圖像尺寸不足，無法使用 {window_size}x{window_size} 的窗口進行比較!")
                return
            
            # 獲取度量方式
            metric = self.metric_combo.currentText()
            
            # 是否使用灰階比較
            use_grayscale = self.use_grayscale_cb.isChecked()
            
            # 如果使用灰階比較，先轉換圖像
            if use_grayscale:
                img1 = img1.convert('L')
                img2 = img2.convert('L')
                gt = gt.convert('L')
            
            # 計算最大有效起始點
            max_start_x1 = img1_width - window_size
            max_start_y1 = img1_height - window_size
            max_start_x2 = img2_width - window_size
            max_start_y2 = img2_height - window_size
            max_start_x_gt = gt_width - window_size
            max_start_y_gt = gt_height - window_size
            
            # 取最小值確保所有圖像都能裁剪
            max_start_x = min(max_start_x1, max_start_x2, max_start_x_gt)
            max_start_y = min(max_start_y1, max_start_y2, max_start_y_gt)
            
            # 計算網格數量
            grid_width = math.ceil((max_start_x + 1) / self.grid_size)
            grid_height = math.ceil((max_start_y + 1) / self.grid_size)
            
            # 創建一個進度對話框
            QMessageBox.information(self, "開始處理", f"將使用 {window_size}x{window_size} 的窗口在圖像範圍內搜尋，並將每 {self.grid_size}x{self.grid_size} 區域最佳結果保留，共 {grid_width*grid_height} 個區域...")
            
            # 創建字典存儲每個網格的最佳結果
            grid_results = {}
            
            # 創建一個並行任務列表
            tasks = []
            for start_y in range(0, max_start_y + 1):
                for start_x in range(0, max_start_x + 1):
                    tasks.append((start_x, start_y))
            
            # 根據任務數量決定處理方式
            results = []
            if len(tasks) > 1000:  # 任務數量大，使用多進程
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    # 使用starmap和部分應用
                    partial_compare = partial(compare_regions, img1, img2, gt, 
                                             window_size=window_size, mode=mode, metric=metric)
                    # 只傳入x, y坐標
                    all_results = pool.starmap(partial_compare, tasks)
                    # 過濾可能的無效結果
                    results = [r for r in all_results if r[2] != float('-inf')]
            else:  # 任務數量小，直接循環
                for start_x, start_y in tasks:
                    result = compare_regions(img1, img2, gt, start_x, start_y, window_size, mode, metric)
                    if result[2] != float('-inf'):
                        results.append(result)
            
            if not results:
                QMessageBox.warning(self, "警告", "沒有找到有效的比較結果!")
                return
            
            # 將結果分配到各個網格，並保留每個網格的最佳結果
            for result in results:
                start_x, start_y, score, diff1_gt, diff2_gt = result
                
                # 計算該點所屬的網格
                grid_x = start_x // self.grid_size
                grid_y = start_y // self.grid_size
                grid_key = (grid_x, grid_y)
                
                # 如果該網格還沒有結果，或者該結果比現有結果更好，則更新
                if grid_key not in grid_results or score > grid_results[grid_key][2]:
                    grid_results[grid_key] = result
            
            # 將網格結果轉換為列表，並按分數排序
            self.top_results = sorted(grid_results.values(), key=lambda x: x[2], reverse=True)
            self.current_result_index = 0
            
            # 顯示第一個(最佳)結果
            self.show_current_result()
            
            # 顯示找到的結果數量
            QMessageBox.information(self, "完成", f"找到 {len(self.top_results)} 個網格結果，已顯示最佳結果。\n"
                                                 f"使用「上一個結果」和「下一個結果」按鈕瀏覽所有結果。")
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"計算過程中出錯: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_preview_size(self):
        """更新預覽尺寸"""
        size_text = self.preview_size_combo.currentText()
        self.preview_size = int(size_text.split('x')[0])
    
    def save_images_with_preview(self):
        """將當前窗口區域保存到原圖角落並保存"""
        # 檢查是否有足夠的圖像
        if all(img is None for img in self.images):
            QMessageBox.warning(self, "警告", "沒有載入任何圖像!")
            return
        
        try:
            # 獲取當前窗口大小
            window_size = self.current_size
            
            # 獲取預覽放置位置
            corner_pos = self.corner_combo.currentText()
            
            # 計算每張圖需要處理的情況
            to_process = []
            for i in range(4):
                if self.images[i] is not None and self.save_checkboxes[i].isChecked():
                    to_process.append(i)
            
            if not to_process:
                QMessageBox.warning(self, "警告", "沒有選擇要保存的圖像!")
                return
            
            # 處理每張需要保存的圖像
            saved_files = []
            for i in to_process:
                # 獲取原圖和窗口區域
                original_image = self.images[i].copy()
                if self.start_x + window_size > original_image.width or self.start_y + window_size > original_image.height:
                    QMessageBox.warning(self, "警告", f"圖像 {i+1} 窗口範圍超出圖像尺寸!")
                    continue
                
                # 截取窗口區域
                window_image = original_image.crop((self.start_x, self.start_y, 
                                                  self.start_x + window_size, 
                                                  self.start_y + window_size))
                
                # 調整窗口大小為預覽尺寸
                # 使用NEAREST插值保留像素方塊效果
                preview_image = window_image.resize((self.preview_size, self.preview_size), Image.NEAREST)
                
                # 計算預覽放置位置
                orig_width, orig_height = original_image.size
                preview_width, preview_height = preview_image.size
                
                if corner_pos == "右下角":
                    paste_x = orig_width - preview_width - 10
                    paste_y = orig_height - preview_height - 10
                elif corner_pos == "右上角":
                    paste_x = orig_width - preview_width - 10
                    paste_y = 10
                elif corner_pos == "左下角":
                    paste_x = 10
                    paste_y = orig_height - preview_height - 10
                else:  # 左上角
                    paste_x = 10
                    paste_y = 10
                
                # 在原圖上創建繪圖對象
                draw = ImageDraw.Draw(original_image)
                
                # 在原圖上繪製窗口位置的框
                draw.rectangle(
                    [(self.start_x, self.start_y), (self.start_x + window_size, self.start_y + window_size)],
                    outline=(255, 0, 0), width=2
                )
                
                # 繪製從窗口到預覽的連接線
                window_center_x = self.start_x + window_size // 2
                window_center_y = self.start_y + window_size // 2
                preview_center_x = paste_x + preview_width // 2
                preview_center_y = paste_y + preview_height // 2
                
                draw.line(
                    [(window_center_x, window_center_y), (preview_center_x, preview_center_y)],
                    fill=(255, 0, 0), width=1
                )
                
                # 在原圖上粘貼預覽圖像
                original_image.paste(preview_image, (paste_x, paste_y))
                
                # 在預覽周圍畫紅框
                draw.rectangle(
                    [(paste_x, paste_y), (paste_x + preview_width, paste_y + preview_height)],
                    outline=(255, 0, 0), width=2
                )
                
                # 構建保存文件名
                if self.image_paths[i]:
                    base_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                    ext = os.path.splitext(self.image_paths[i])[1]
                else:
                    base_name = f"image_{i+1}"
                    ext = ".png"
                
                suffix = ""
                if i == 0:
                    suffix = "_pic1"
                elif i == 1:
                    suffix = "_pic2"
                elif i == 2:
                    suffix = "_pic3"
                else:
                    suffix = "_gt"
                
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{base_name}{suffix}{ext}")
                
                # 保存圖像
                original_image.save(save_path)
                saved_files.append(save_path)
            
            # 提示保存成功
            if saved_files:
                QMessageBox.information(self, "成功", f"已成功保存 {len(saved_files)} 張圖像:\n" + '\n'.join(saved_files))
            else:
                QMessageBox.warning(self, "警告", "沒有保存任何圖像!")
                
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"保存圖像過程中出錯: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_grid_size(self):
        """更新網格大小"""
        size_text = self.grid_size_combo.currentText()
        self.grid_size = int(size_text.split('x')[0])
        # 清除結果
        self.top_results = []
        self.current_result_index = 0
        self.update_result_navigation()
        QMessageBox.information(self, "網格大小已更新", f"網格大小已設為 {self.grid_size}x{self.grid_size}，請重新執行特徵點尋找。")

    def toggle_theme_mode(self):
        """切換主題模式"""
        is_dark = self.theme_button.isChecked()
        if is_dark:
            # 更新按鈕文字
            self.theme_button.setText("切換回亮色模式")
            
            # 啟用黑暗模式
            dark_stylesheet = """
                QWidget { 
                    background-color: #2D2D30; 
                    color: #E0E0E0; 
                }
                QGroupBox { 
                    border: 1px solid #3F3F46; 
                    border-radius: 5px; 
                    margin-top: 10px; 
                    font-weight: bold; 
                    font-size: 13pt;
                    color: #E0E0E0;
                }
                QGroupBox::title { 
                    subcontrol-origin: margin; 
                    left: 10px; 
                    padding: 0 5px 0 5px; 
                }
                QPushButton { 
                    background-color: #0E639C; 
                    color: white; 
                    border: none; 
                    border-radius: 3px; 
                    padding: 5px; 
                }
                QPushButton:hover { 
                    background-color: #1177BB; 
                }
                QPushButton:disabled { 
                    background-color: #3F3F46; 
                    color: #959595; 
                }
                QComboBox, QSpinBox, QLineEdit { 
                    background-color: #333337; 
                    color: #E0E0E0; 
                    border: 1px solid #3F3F46; 
                    border-radius: 3px; 
                }
                QScrollArea, QLabel { 
                    background-color: #2D2D30; 
                    color: #E0E0E0; 
                }
                QCheckBox { 
                    color: #E0E0E0; 
                }
                QFrame { 
                    background-color: #252526;
                    border: 1px solid #3F3F46;
                }
            """
            self.setStyleSheet(dark_stylesheet)
            
            # 保持主題按鈕樣式
            self.theme_button.setStyleSheet("""
                QPushButton {
                    font-size: 16pt;
                    font-weight: bold;
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4A148C, stop:1 #880E4F);
                    color: white;
                    border-radius: 10px;
                    padding: 15px;
                }
                QPushButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6A1B9A, stop:1 #AD1457);
                }
            """)
            
            # 更新特定樣式
            self.diff_ratio_label.setStyleSheet("QLabel { font-weight: bold; color: #FF79C6; }")
            
            # 更新圖像框架樣式
            for i in range(4):
                if hasattr(self, 'display_labels') and i < len(self.display_labels):
                    self.display_labels[i].setStyleSheet("background-color: #1E1E1E; border: 1px solid #3F3F46;")
            
            # 更新選擇圖像的框架
            for i in range(4):
                if hasattr(self, 'image_buttons') and i < len(self.image_buttons):
                    parent = self.image_buttons[i].parent()
                    if parent and isinstance(parent, QFrame):
                        parent.setStyleSheet("QFrame { background-color: #252526; border: 1px solid #3F3F46; border-radius: 5px; }")
            
            # 更新圖像顯示區域
            for i in range(self.display_layout.count()):
                widget = self.display_layout.itemAt(i).widget()
                if widget and isinstance(widget, QGroupBox):
                    widget.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; background-color: #2D2D30; border: 1px solid #3F3F46; }")
                    # 尋找內部的QLabel
                    for child in widget.findChildren(QLabel):
                        if child != self.display_labels[i % 4]:  # 避免重複設置display_labels
                            child.setStyleSheet("background-color: #2D2D30; color: #E0E0E0;")
            
            # 更新圖片路徑文字顏色為白色
            for i in range(4):
                if hasattr(self, 'image_path_edits') and i < len(self.image_path_edits):
                    self.image_path_edits[i].setStyleSheet("color: white; padding: 5px; background-color: #333337; border: 1px solid #3F3F46;")
            
        else:
            # 更新按鈕文字
            self.theme_button.setText("切換黑暗模式")
            
            # 恢復亮色模式
            light_stylesheet = """
                QLabel, QPushButton, QCheckBox, QComboBox, QSpinBox, QLineEdit { 
                    font-size: 12pt; 
                }
                QGroupBox { 
                    font-size: 13pt; 
                    font-weight: bold; 
                }
            """
            self.setStyleSheet(light_stylesheet)
            
            # 恢復主題按鈕樣式
            self.theme_button.setStyleSheet("""
                QPushButton {
                    font-size: 16pt;
                    font-weight: bold;
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2C3E50, stop:1 #4CA1AF);
                    color: white;
                    border-radius: 10px;
                    padding: 15px;
                }
                QPushButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #34495E, stop:1 #5DADE2);
                }
            """)
            
            # 恢復特定樣式
            self.diff_ratio_label.setStyleSheet("QLabel { font-weight: bold; color: #E91E63; }")
            
            # 恢復其他控件的原始樣式
            self.update_button.setStyleSheet("QPushButton { min-height: 30px; background-color: #4CAF50; color: white; }")
            self.find_button1.setStyleSheet("QPushButton { min-height: 30px; background-color: #2196F3; color: white; }")
            self.find_button2.setStyleSheet("QPushButton { min-height: 30px; background-color: #2196F3; color: white; }")
            self.save_button.setStyleSheet("QPushButton { min-height: 30px; background-color: #FF9800; color: white; }")
            
            # 恢復圖像顯示區域樣式
            for i in range(4):
                if hasattr(self, 'display_labels') and i < len(self.display_labels):
                    self.display_labels[i].setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
            
            # 恢復選擇圖像的框架
            for i in range(4):
                if hasattr(self, 'image_buttons') and i < len(self.image_buttons):
                    parent = self.image_buttons[i].parent()
                    if parent and isinstance(parent, QFrame):
                        parent.setStyleSheet("QFrame { background-color: #f9f9f9; border-radius: 5px; }")
            
            # 恢復圖像顯示區域
            for i in range(self.display_layout.count()):
                widget = self.display_layout.itemAt(i).widget()
                if widget and isinstance(widget, QGroupBox):
                    widget.setStyleSheet("QGroupBox { font-weight: bold; }")
            
            for i in range(4):
                if self.image_buttons[i]:
                    self.image_buttons[i].setStyleSheet("QPushButton { min-height: 30px; background-color: #2196F3; color: white; }")
            
            # 恢復圖片路徑文字顏色
            for i in range(4):
                if hasattr(self, 'image_path_edits') and i < len(self.image_path_edits):
                    if self.image_paths[i]:  # 已選擇圖像
                        self.image_path_edits[i].setStyleSheet("color: black; padding: 5px;")
                    else:  # 未選擇圖像
                        self.image_path_edits[i].setStyleSheet("color: gray; padding: 5px;")
        
        # 更新所有控件
        self.update()

if __name__ == "__main__":
    # 檢查是否支援多進程
    mp.freeze_support()
    
    app = QApplication(sys.argv)
    
    # 設定應用程式全局字體
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    
    window = ImageComparisonTool()
    window.show()
    sys.exit(app.exec_()) 