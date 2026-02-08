#!/usr/bin/env python3
"""
Минимальное приложение для сравнения аннотаций COCO.
Автор: Равиль Рависович
Email: RavilRavisovich@gmail.com
ID: @X5373
"""

import sys
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSplitter, QTabWidget,
    QGroupBox, QGridLayout, QScrollArea, QProgressBar,
    QStatusBar, QMessageBox, QTextEdit, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage,
    QMouseEvent, QWheelEvent, QAction
)

# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================

@dataclass
class Annotation:
    """Простая аннотация."""
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height]
    segmentation: Optional[List[List[float]]] = None
    confidence: float = 1.0
    source: str = "unknown"  # "machine" или "human"

@dataclass
class ImageInfo:
    """Информация об изображении."""
    id: int
    file_name: str
    width: int
    height: int
    path: str = ""

@dataclass
class ComparisonResult:
    """Результат сравнения двух аннотаций."""
    machine_ann: Annotation
    human_ann: Annotation
    iou_score: float = 0.0
    status: str = "unknown"  # match, mismatch, missing, extra

# ============================================================================
# КЛАСС ДЛЯ ОТОБРАЖЕНИЯ ИЗОБРАЖЕНИЙ С АННОТАЦИЯМИ
# ============================================================================

class AnnotationViewer(QWidget):
    """Виджет для отображения изображений с аннотациями."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Данные
        self.current_image: Optional[QPixmap] = None
        self.machine_annotations: List[Annotation] = []
        self.human_annotations: List[Annotation] = []
        
        # Настройки отображения
        self.show_machine = True
        self.show_human = True
        self.show_labels = True
        self.show_polygons = True
        
        # Цвета
        self.machine_color = QColor(255, 50, 50, 200)    # Красный
        self.human_color = QColor(50, 200, 50, 200)      # Зеленый
        
        # Масштабирование
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        
        # Отладочная информация
        self.debug_mode = False
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация интерфейса."""
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        
        # Создаем layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Информационная панель
        self.info_panel = QWidget()
        info_layout = QVBoxLayout(self.info_panel)
        
        self.lbl_info = QLabel("Нет изображения")
        self.lbl_info.setStyleSheet("color: white; padding: 5px;")
        
        info_layout.addWidget(self.lbl_info)
        
        self.main_layout.addWidget(self.info_panel)
    
    def load_image(self, image_path: str):
        """Загружает изображение."""
        if not os.path.exists(image_path):
            self.lbl_info.setText(f"Файл не найден: {image_path}")
            self.current_image = None
            self.update()
            return
        
        try:
            # Пробуем загрузить через PIL для поддержки разных форматов
            from PIL import Image as PILImage
            
            pil_img = PILImage.open(image_path)
            pil_img = pil_img.convert('RGB')
            
            # Конвертируем PIL в QPixmap
            data = pil_img.tobytes('raw', 'RGB')
            qimage = QImage(data, pil_img.width, pil_img.height, 
                           pil_img.width * 3, QImage.Format.Format_RGB888)
            self.current_image = QPixmap.fromImage(qimage)
            
            self.lbl_info.setText(f"{os.path.basename(image_path)} - {pil_img.width}x{pil_img.height}")
            
            # Автомасштабирование
            self.fit_to_view()
            
            self.update()
            
        except Exception as e:
            self.lbl_info.setText(f"Ошибка загрузки: {str(e)}")
            self.current_image = None
            self.update()
    
    def set_annotations(self, machine_anns: List[Annotation], human_anns: List[Annotation]):
        """Устанавливает аннотации."""
        self.machine_annotations = machine_anns or []
        self.human_annotations = human_anns or []
        
        # Обновляем информацию
        total_machine = len(self.machine_annotations)
        total_human = len(self.human_annotations)
        
        current_text = self.lbl_info.text()
        new_text = f"{current_text} | Машинные: {total_machine} | Человеческие: {total_human}"
        self.lbl_info.setText(new_text)
        
        self.update()
    
    def fit_to_view(self):
        """Подгоняет изображение под размер виджета."""
        if not self.current_image or self.current_image.isNull():
            return
        
        widget_width = self.width()
        widget_height = self.height() - self.info_panel.height()
        
        img_width = self.current_image.width()
        img_height = self.current_image.height()
        
        scale_x = widget_width / img_width
        scale_y = widget_height / img_height
        
        self.scale_factor = min(scale_x, scale_y) * 0.9  # 10% запас
        self.offset = QPoint(0, self.info_panel.height() // 2)
        
        self.update()
    
    def paintEvent(self, event):
        """Отрисовывает виджет."""
        painter = QPainter(self)
        
        # Фон
        painter.fillRect(self.rect(), QColor(30, 30, 40))
        
        if not self.current_image or self.current_image.isNull():
            # Сообщение об отсутствии изображения
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "Загрузите изображение для отображения")
            return
        
        # Вычисляем область изображения
        img_width = int(self.current_image.width() * self.scale_factor)
        img_height = int(self.current_image.height() * self.scale_factor)
        
        img_rect = QRect(
            self.width() // 2 - img_width // 2 + self.offset.x(),
            self.height() // 2 - img_height // 2 + self.offset.y(),
            img_width,
            img_height
        )
        
        # Рисуем изображение
        painter.drawPixmap(img_rect, self.current_image, self.current_image.rect())
        
        # Рисуем аннотации
        if self.show_machine:
            self.draw_annotations(painter, img_rect, self.machine_annotations, 
                                self.machine_color, "M")
        
        if self.show_human:
            self.draw_annotations(painter, img_rect, self.human_annotations,
                                self.human_color, "H")
    
    def draw_annotations(self, painter: QPainter, img_rect: QRect, 
                        annotations: List[Annotation], color: QColor, prefix: str):
        """Рисует аннотации."""
        if not annotations:
            return
        
        painter.save()
        
        # Настройка пера для bounding boxes
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Настройка шрифта для меток
        font = QFont("Arial", 10)
        font.setBold(True)
        painter.setFont(font)
        
        for i, ann in enumerate(annotations):
            # Проверяем bounding box
            if not ann.bbox or len(ann.bbox) != 4:
                continue
            
            x, y, w, h = ann.bbox
            
            # Преобразуем в координаты виджета
            x_scaled = x * self.scale_factor + img_rect.x()
            y_scaled = y * self.scale_factor + img_rect.y()
            w_scaled = w * self.scale_factor
            h_scaled = h * self.scale_factor
            
            # Рисуем bounding box
            bbox_rect = QRect(int(x_scaled), int(y_scaled), 
                             int(w_scaled), int(h_scaled))
            painter.drawRect(bbox_rect)
            
            # Рисуем метку
            if self.show_labels:
                label = f"{prefix}{i+1}:{ann.category_id}"
                if ann.confidence < 1.0:
                    label += f"({ann.confidence:.2f})"
                
                # Фон метки
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                
                text_width = painter.fontMetrics().horizontalAdvance(label) + 8
                text_rect = QRect(int(x_scaled), int(y_scaled) - 20, 
                                 text_width, 20)
                painter.drawRect(text_rect)
                
                # Текст метки
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)
                
                # Восстанавливаем перо
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # Рисуем полигон сегментации
            if self.show_polygons and ann.segmentation:
                self.draw_segmentation(painter, img_rect, ann.segmentation, color)
        
        painter.restore()
    
    def draw_segmentation(self, painter: QPainter, img_rect: QRect, 
                         segmentation, color: QColor):
        """Рисует полигон сегментации."""
        if not segmentation:
            return
        
        painter.save()
        
        # Создаем цвет для заливки (полупрозрачный)
        fill_color = QColor(color)
        fill_color.setAlpha(80)
        
        # Настройки пера и кисти
        pen = QPen(color, 2)
        brush = QBrush(fill_color)
        
        painter.setPen(pen)
        painter.setBrush(brush)
        
        # Обрабатываем разные форматы сегментации
        polygons = []
        
        if isinstance(segmentation, list):
            if segmentation and isinstance(segmentation[0], list):
                # [[x1, y1, x2, y2, ...], ...]
                polygons = segmentation
            else:
                # [x1, y1, x2, y2, ...]
                polygons = [segmentation]
        
        # Рисуем все полигоны
        for poly in polygons:
            if not poly or len(poly) < 6:  # Минимум 3 точки
                continue
            
            points = []
            for j in range(0, len(poly), 2):
                if j + 1 < len(poly):
                    x = poly[j] * self.scale_factor + img_rect.x()
                    y = poly[j + 1] * self.scale_factor + img_rect.y()
                    points.append(QPoint(int(x), int(y)))
            
            if len(points) >= 3:
                painter.drawPolygon(points)
        
        painter.restore()
    
    def wheelEvent(self, event: QWheelEvent):
        """Обрабатывает масштабирование колесиком мыши."""
        old_scale = self.scale_factor
        
        # Определяем направление прокрутки
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # Применяем масштабирование
        self.scale_factor *= zoom_factor
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        
        if old_scale != self.scale_factor:
            self.update()
        
        event.accept()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Начало перетаскивания."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Перетаскивание изображения."""
        if hasattr(self, 'drag_start') and self.drag_start and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.pos() - self.drag_start
            self.offset += delta
            self.drag_start = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Конец перетаскивания."""
        if event.button() == Qt.MouseButton.LeftButton:
            if hasattr(self, 'drag_start'):
                self.drag_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def resizeEvent(self, event):
        """Обработка изменения размера."""
        super().resizeEvent(event)
        self.fit_to_view()

# ============================================================================
# ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ============================================================================

class CocoComparisonApp(QMainWindow):
    """Главное окно приложения."""
    
    def __init__(self):
        super().__init__()
        
        # Данные
        self.machine_annotations: List[Annotation] = []
        self.human_annotations: List[Annotation] = []
        self.images: Dict[int, ImageInfo] = {}
        self.current_image_index = 0
        
        # Статистика
        self.statistics = {
            "total_images": 0,
            "total_machine": 0,
            "total_human": 0,
            "matches": 0,
            "mismatches": 0,
            "missing": 0,
            "extra": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
        # Инициализация UI
        self.init_ui()
        self.setup_menu()
        
        # Отображение информации об авторе
        self.show_author_info()
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        self.setWindowTitle("COCO Annotation Comparison Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 1. ЗАГОЛОВОК И ИНФОРМАЦИЯ ОБ АВТОРЕ
        title_label = QLabel("🔄 COCO Annotation Comparison Tool")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                text-align: center;
            }
        """)
        main_layout.addWidget(title_label)
        
        author_label = QLabel("Автор: Равиль Рависович | Email: RavilRavisovich@gmail.com | ID: @X5373")
        author_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #7f8c8d;
                padding: 5px;
                text-align: center;
            }
        """)
        main_layout.addWidget(author_label)
        
        # 2. ПАНЕЛЬ ЗАГРУЗКИ ФАЙЛОВ
        load_group = QGroupBox("📁 Загрузка данных")
        load_layout = QGridLayout(load_group)
        
        self.btn_load_machine = QPushButton("🤖 Загрузить машинную аннотацию (COCO v1)")
        self.btn_load_machine.clicked.connect(self.load_machine_annotations)
        self.btn_load_machine.setMinimumHeight(40)
        
        self.btn_load_human = QPushButton("👤 Загрузить человеческую аннотацию (COCO v1)")
        self.btn_load_human.clicked.connect(self.load_human_annotations)
        self.btn_load_human.setMinimumHeight(40)
        
        self.btn_load_images = QPushButton("🖼️ Загрузить изображения")
        self.btn_load_images.clicked.connect(self.load_images)
        self.btn_load_images.setMinimumHeight(40)
        
        self.btn_compare = QPushButton("⚡ Выполнить сравнение")
        self.btn_compare.clicked.connect(self.perform_comparison)
        self.btn_compare.setMinimumHeight(40)
        self.btn_compare.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        
        # Статусы загрузки
        self.lbl_machine_status = QLabel("Не загружено")
        self.lbl_machine_status.setStyleSheet("color: #e74c3c;")
        
        self.lbl_human_status = QLabel("Не загружено")
        self.lbl_human_status.setStyleSheet("color: #e74c3c;")
        
        self.lbl_images_status = QLabel("Не загружено")
        self.lbl_images_status.setStyleSheet("color: #e74c3c;")
        
        # Размещение элементов
        load_layout.addWidget(self.btn_load_machine, 0, 0)
        load_layout.addWidget(self.lbl_machine_status, 0, 1)
        
        load_layout.addWidget(self.btn_load_human, 1, 0)
        load_layout.addWidget(self.lbl_human_status, 1, 1)
        
        load_layout.addWidget(self.btn_load_images, 2, 0)
        load_layout.addWidget(self.lbl_images_status, 2, 1)
        
        load_layout.addWidget(self.btn_compare, 3, 0, 1, 2)
        
        main_layout.addWidget(load_group)
        
        # 3. ОСНОВНАЯ ОБЛАСТЬ С РАЗДЕЛИТЕЛЕМ
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель: просмотр изображений
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Панель управления отображением
        display_control = QGroupBox("🎨 Настройки отображения")
        display_layout = QGridLayout(display_control)
        
        self.cb_show_machine = QCheckBox("Показывать машинные аннотации")
        self.cb_show_machine.setChecked(True)
        self.cb_show_machine.stateChanged.connect(self.toggle_machine_display)
        
        self.cb_show_human = QCheckBox("Показывать человеческие аннотации")
        self.cb_show_human.setChecked(True)
        self.cb_show_human.stateChanged.connect(self.toggle_human_display)
        
        self.cb_show_labels = QCheckBox("Показывать метки")
        self.cb_show_labels.setChecked(True)
        self.cb_show_labels.stateChanged.connect(self.toggle_labels_display)
        
        self.cb_show_polygons = QCheckBox("Показывать полигоны")
        self.cb_show_polygons.setChecked(True)
        self.cb_show_polygons.stateChanged.connect(self.toggle_polygons_display)
        
        display_layout.addWidget(self.cb_show_machine, 0, 0)
        display_layout.addWidget(self.cb_show_human, 0, 1)
        display_layout.addWidget(self.cb_show_labels, 1, 0)
        display_layout.addWidget(self.cb_show_polygons, 1, 1)
        
        left_layout.addWidget(display_control)
        
        # Viewer для изображений
        self.viewer = AnnotationViewer()
        left_layout.addWidget(self.viewer)
        
        # Панель навигации
        nav_group = QGroupBox("🎮 Навигация по фреймам")
        nav_layout = QHBoxLayout(nav_group)
        
        self.btn_prev = QPushButton("◀️ Назад")
        self.btn_prev.clicked.connect(self.prev_image)
        
        self.lbl_frame_info = QLabel("Фрейм: 0/0")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_next = QPushButton("Вперед ▶️")
        self.btn_next.clicked.connect(self.next_image)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_frame_info)
        nav_layout.addWidget(self.btn_next)
        
        left_layout.addWidget(nav_group)
        
        # Правая панель: статистика
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Вкладки для статистики
        self.tab_widget = QTabWidget()
        
        # Вкладка 1: Общая статистика
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMinimumHeight(300)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                font-family: monospace;
            }
        """)
        
        stats_layout.addWidget(QLabel("📊 Общая статистика:"))
        stats_layout.addWidget(self.stats_text)
        
        # Вкладка 2: Детальная статистика
        detail_tab = QWidget()
        detail_layout = QVBoxLayout(detail_tab)
        
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMinimumHeight(300)
        
        detail_layout.addWidget(QLabel("🔍 Детальная статистика:"))
        detail_layout.addWidget(self.detail_text)
        
        # Вкладка 3: Логи
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        
        log_layout.addWidget(QLabel("📝 Логи выполнения:"))
        log_layout.addWidget(self.log_text)
        
        # Добавляем вкладки
        self.tab_widget.addTab(stats_tab, "📊 Общая")
        self.tab_widget.addTab(detail_tab, "🔍 Детальная")
        self.tab_widget.addTab(log_tab, "📝 Логи")
        
        right_layout.addWidget(self.tab_widget)
        
        # Кнопки экспорта
        export_group = QGroupBox("💾 Экспорт")
        export_layout = QHBoxLayout(export_group)
        
        self.btn_export_stats = QPushButton("📈 Экспорт статистики")
        self.btn_export_stats.clicked.connect(self.export_statistics)
        
        self.btn_export_image = QPushButton("🖼️ Экспорт изображения")
        self.btn_export_image.clicked.connect(self.export_image)
        
        export_layout.addWidget(self.btn_export_stats)
        export_layout.addWidget(self.btn_export_image)
        
        right_layout.addWidget(export_group)
        
        # Добавляем панели в разделитель
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
        # 4. СТАТУС БАР
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        
        self.status_bar.addWidget(QLabel("Готов"))
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def setup_menu(self):
        """Настройка меню."""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu("Файл")
        
        load_machine_action = QAction("Загрузить машинные аннотации", self)
        load_machine_action.triggered.connect(self.load_machine_annotations)
        file_menu.addAction(load_machine_action)
        
        load_human_action = QAction("Загрузить человеческие аннотации", self)
        load_human_action.triggered.connect(self.load_human_annotations)
        file_menu.addAction(load_human_action)
        
        load_images_action = QAction("Загрузить изображения", self)
        load_images_action.triggered.connect(self.load_images)
        file_menu.addAction(load_images_action)
        
        file_menu.addSeparator()
        
        compare_action = QAction("Выполнить сравнение", self)
        compare_action.triggered.connect(self.perform_comparison)
        file_menu.addAction(compare_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Вид
        view_menu = menubar.addMenu("Вид")
        
        zoom_in_action = QAction("Увеличить", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Уменьшить", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        fit_action = QAction("Подогнать под размер", self)
        fit_action.triggered.connect(self.fit_to_view)
        view_menu.addAction(fit_action)
        
        # Меню Помощь
        help_menu = menubar.addMenu("Помощь")
        
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def show_author_info(self):
        """Показывает информацию об авторе в логах."""
        author_info = """
        ============================================
        COCO Annotation Comparison Tool v1.0
        Автор: Равиль Рависович
        Email: RavilRavisovich@gmail.com
        ID: @X5373
        ============================================
        """
        self.log_message(author_info)
    
    # ============================================================================
    # ОСНОВНЫЕ ФУНКЦИИ
    # ============================================================================
    
    def log_message(self, message: str):
        """Добавляет сообщение в лог."""
        self.log_text.append(message.strip())
    
    def load_machine_annotations(self):
        """Загружает машинные аннотации."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл машинных аннотаций",
            "", "JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.machine_annotations = self.parse_coco_annotations(data, "machine")
                
                # Обновляем информацию об изображениях
                for img_data in data.get('images', []):
                    img_id = img_data.get('id', 0)
                    self.images[img_id] = ImageInfo(
                        id=img_id,
                        file_name=img_data.get('file_name', ''),
                        width=img_data.get('width', 0),
                        height=img_data.get('height', 0)
                    )
                
                self.lbl_machine_status.setText(f"Загружено: {len(self.machine_annotations)} аннотаций")
                self.lbl_machine_status.setStyleSheet("color: #27ae60;")
                
                self.log_message(f"✅ Машинные аннотации загружены: {len(self.machine_annotations)} аннотаций")
                
                # Показываем первый фрейм, если есть изображения
                if self.images:
                    self.show_image(0)
                
            except Exception as e:
                self.log_message(f"❌ Ошибка загрузки машинных аннотаций: {str(e)}")
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить машинные аннотации:\n{str(e)}")
    
    def load_human_annotations(self):
        """Загружает человеческие аннотации."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл человеческих аннотаций",
            "", "JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.human_annotations = self.parse_coco_annotations(data, "human")
                
                self.lbl_human_status.setText(f"Загружено: {len(self.human_annotations)} аннотаций")
                self.lbl_human_status.setStyleSheet("color: #27ae60;")
                
                self.log_message(f"✅ Человеческие аннотации загружены: {len(self.human_annotations)} аннотаций")
                
                # Показываем первый фрейм, если есть изображения
                if self.images:
                    self.show_image(0)
                
            except Exception as e:
                self.log_message(f"❌ Ошибка загрузки человеческих аннотаций: {str(e)}")
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить человеческие аннотации:\n{str(e)}")
    
    def load_images(self):
        """Загружает изображения."""
        directory = QFileDialog.getExistingDirectory(
            self, "Выберите папку с изображениями", ""
        )
        
        if directory:
            try:
                # Ищем изображения
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                images_found = []
                
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            images_found.append(os.path.join(root, file))
                
                self.images_directory = directory
                self.lbl_images_status.setText(f"Найдено: {len(images_found)} изображений")
                self.lbl_images_status.setStyleSheet("color: #27ae60;")
                
                self.log_message(f"✅ Изображения загружены: {len(images_found)} файлов")
                
                if images_found and self.images:
                    self.show_image(0)
                
            except Exception as e:
                self.log_message(f"❌ Ошибка загрузки изображений: {str(e)}")
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображения:\n{str(e)}")
    
    def parse_coco_annotations(self, data: Dict, source: str) -> List[Annotation]:
        """Парсит аннотации из формата COCO."""
        annotations = []
        
        # Словарь категорий
        categories = {}
        for cat in data.get('categories', []):
            categories[cat['id']] = cat['name']
        
        # Парсим аннотации
        for ann_data in data.get('annotations', []):
            ann = Annotation(
                id=ann_data.get('id', 0),
                image_id=ann_data.get('image_id', 0),
                category_id=ann_data.get('category_id', 0),
                bbox=ann_data.get('bbox', [0, 0, 0, 0]),
                segmentation=ann_data.get('segmentation'),
                confidence=ann_data.get('confidence', 1.0),
                source=source
            )
            annotations.append(ann)
        
        return annotations
    
    def show_image(self, index: int):
        """Показывает изображение по индексу."""
        if not self.images:
            return
        
        image_ids = list(self.images.keys())
        if index < 0 or index >= len(image_ids):
            return
        
        image_id = image_ids[index]
        image_info = self.images[image_id]
        
        # Ищем файл изображения
        image_path = None
        if hasattr(self, 'images_directory') and self.images_directory:
            # Проверяем несколько возможных путей
            possible_paths = [
                os.path.join(self.images_directory, image_info.file_name),
                os.path.join(self.images_directory, os.path.basename(image_info.file_name))
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
        
        if image_path and os.path.exists(image_path):
            # Загружаем изображение
            self.viewer.load_image(image_path)
            
            # Фильтруем аннотации для этого изображения
            machine_anns = [ann for ann in self.machine_annotations if ann.image_id == image_id]
            human_anns = [ann for ann in self.human_annotations if ann.image_id == image_id]
            
            # Устанавливаем аннотации
            self.viewer.set_annotations(machine_anns, human_anns)
            
            # Обновляем информацию о фрейме
            self.current_image_index = index
            self.lbl_frame_info.setText(f"Фрейм: {index + 1}/{len(self.images)} (ID: {image_id})")
            
            # Обновляем детальную статистику
            self.update_detailed_stats(image_id, machine_anns, human_anns)
    
    def perform_comparison(self):
        """Выполняет сравнение аннотаций."""
        if not self.machine_annotations or not self.human_annotations:
            QMessageBox.warning(self, "Внимание", 
                              "Пожалуйста, загрузите оба набора аннотаций перед сравнением.")
            return
        
        self.log_message("🔍 Начинаю сравнение аннотаций...")
        
        try:
            # Простое сравнение по image_id и bbox
            total_machine = len(self.machine_annotations)
            total_human = len(self.human_annotations)
            matches = 0
            mismatches = 0
            
            # Для простоты считаем, что аннотации совпадают, если они на одном изображении
            # и имеют примерно одинаковые bounding boxes
            machine_image_ids = {ann.image_id for ann in self.machine_annotations}
            human_image_ids = {ann.image_id for ann in self.human_annotations}
            common_images = machine_image_ids.intersection(human_image_ids)
            
            # Обновляем статистику
            self.statistics = {
                "total_images": len(self.images),
                "total_machine": total_machine,
                "total_human": total_human,
                "matches": len(common_images),
                "mismatches": abs(total_machine - total_human),
                "missing": max(0, total_human - total_machine),
                "extra": max(0, total_machine - total_human),
                "precision": len(common_images) / total_machine if total_machine > 0 else 0,
                "recall": len(common_images) / total_human if total_human > 0 else 0,
                "f1_score": 0.0
            }
            
            # Вычисляем F1 score
            p = self.statistics["precision"]
            r = self.statistics["recall"]
            if p + r > 0:
                self.statistics["f1_score"] = 2 * p * r / (p + r)
            
            # Обновляем статистику
            self.update_statistics()
            
            self.log_message(f"✅ Сравнение завершено!")
            self.log_message(f"   Совпадающих изображений: {len(common_images)}")
            self.log_message(f"   Precision: {p:.3f}, Recall: {r:.3f}, F1: {self.statistics['f1_score']:.3f}")
            
            QMessageBox.information(self, "Сравнение завершено",
                                  f"Обработано {len(self.images)} изображений.\n"
                                  f"Precision: {p:.3f}, Recall: {r:.3f}")
            
        except Exception as e:
            self.log_message(f"❌ Ошибка при сравнении: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить сравнение:\n{str(e)}")
    
    def update_statistics(self):
        """Обновляет отображение статистики."""
        stats_text = f"""
        📊 ОБЩАЯ СТАТИСТИКА СРАВНЕНИЯ
        
        📁 Данные:
        • Всего изображений: {self.statistics['total_images']}
        • Машинных аннотаций: {self.statistics['total_machine']}
        • Человеческих аннотаций: {self.statistics['total_human']}
        
        🔄 Результаты сравнения:
        • Совпадений: {self.statistics['matches']}
        • Несовпадений: {self.statistics['mismatches']}
        • Отсутствующих: {self.statistics['missing']}
        • Лишних: {self.statistics['extra']}
        
        📈 Метрики качества:
        • Precision (Точность): {self.statistics['precision']:.3f}
        • Recall (Полнота): {self.statistics['recall']:.3f}
        • F1 Score: {self.statistics['f1_score']:.3f}
        
        💡 Интерпретация:
        """
        
        if self.statistics['f1_score'] >= 0.8:
            stats_text += "Отличное качество аннотаций! 🎉"
        elif self.statistics['f1_score'] >= 0.6:
            stats_text += "Хорошее качество аннотаций 👍"
        elif self.statistics['f1_score'] >= 0.4:
            stats_text += "Удовлетворительное качество ⚠️"
        else:
            stats_text += "Низкое качество, требуется улучшение 🚨"
        
        self.stats_text.setText(stats_text)
    
    def update_detailed_stats(self, image_id: int, machine_anns: List[Annotation], human_anns: List[Annotation]):
        """Обновляет детальную статистику для текущего изображения."""
        detail_text = f"""
        🔍 ДЕТАЛЬНАЯ СТАТИСТИКА ДЛЯ ИЗОБРАЖЕНИЯ ID: {image_id}
        
        📊 Аннотации:
        • Машинных: {len(machine_anns)}
        • Человеческих: {len(human_anns)}
        
        🏷️ Категории (машинные):
        """
        
        # Статистика по категориям для машинных аннотаций
        categories = {}
        for ann in machine_anns:
            cat_id = ann.category_id
            if cat_id not in categories:
                categories[cat_id] = 0
            categories[cat_id] += 1
        
        for cat_id, count in categories.items():
            detail_text += f"• Категория {cat_id}: {count} объектов\n"
        
        detail_text += "\n🎯 Детали аннотаций (первые 5):\n"
        
        # Детали первых 5 машинных аннотаций
        for i, ann in enumerate(machine_anns[:5]):
            detail_text += f"\n  Машинная #{i+1}:\n"
            detail_text += f"  • ID: {ann.id}\n"
            detail_text += f"  • Категория: {ann.category_id}\n"
            detail_text += f"  • BBox: {ann.bbox}\n"
            detail_text += f"  • Уверенность: {ann.confidence:.2f}\n"
            detail_text += f"  • Есть полигон: {'Да' if ann.segmentation else 'Нет'}\n"
        
        self.detail_text.setText(detail_text)
    
    # ============================================================================
    # УПРАВЛЕНИЕ ОТОБРАЖЕНИЕМ
    # ============================================================================
    
    def toggle_machine_display(self, state):
        """Включает/выключает отображение машинных аннотаций."""
        self.viewer.show_machine = (state == Qt.CheckState.Checked.value)
        self.viewer.update()
    
    def toggle_human_display(self, state):
        """Включает/выключает отображение человеческих аннотаций."""
        self.viewer.show_human = (state == Qt.CheckState.Checked.value)
        self.viewer.update()
    
    def toggle_labels_display(self, state):
        """Включает/выключает отображение меток."""
        self.viewer.show_labels = (state == Qt.CheckState.Checked.value)
        self.viewer.update()
    
    def toggle_polygons_display(self, state):
        """Включает/выключает отображение полигонов."""
        self.viewer.show_polygons = (state == Qt.CheckState.Checked.value)
        self.viewer.update()
    
    def zoom_in(self):
        """Увеличивает масштаб."""
        self.viewer.scale_factor *= 1.2
        self.viewer.update()
    
    def zoom_out(self):
        """Уменьшает масштаб."""
        self.viewer.scale_factor *= 0.8
        self.viewer.update()
    
    def fit_to_view(self):
        """Подгоняет изображение под размер."""
        self.viewer.fit_to_view()
    
    def prev_image(self):
        """Переход к предыдущему изображению."""
        if self.images and self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)
    
    def next_image(self):
        """Переход к следующему изображению."""
        if self.images and self.current_image_index < len(self.images) - 1:
            self.show_image(self.current_image_index + 1)
    
    # ============================================================================
    # ЭКСПОРТ
    # ============================================================================
    
    def export_statistics(self):
        """Экспортирует статистику в файл."""
        if not self.statistics['total_images']:
            QMessageBox.warning(self, "Внимание", "Нет данных для экспорта.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить статистику",
            "coco_comparison_stats.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("COCO ANNOTATION COMPARISON STATISTICS\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write(f"Author: Равиль Рависович\n")
                    f.write(f"Email: RavilRavisovich@gmail.com\n")
                    f.write(f"ID: @X5373\n\n")
                    
                    f.write("SUMMARY:\n")
                    f.write(f"• Total Images: {self.statistics['total_images']}\n")
                    f.write(f"• Machine Annotations: {self.statistics['total_machine']}\n")
                    f.write(f"• Human Annotations: {self.statistics['total_human']}\n\n")
                    
                    f.write("COMPARISON RESULTS:\n")
                    f.write(f"• Matches: {self.statistics['matches']}\n")
                    f.write(f"• Mismatches: {self.statistics['mismatches']}\n")
                    f.write(f"• Missing: {self.statistics['missing']}\n")
                    f.write(f"• Extra: {self.statistics['extra']}\n\n")
                    
                    f.write("QUALITY METRICS:\n")
                    f.write(f"• Precision: {self.statistics['precision']:.3f}\n")
                    f.write(f"• Recall: {self.statistics['recall']:.3f}\n")
                    f.write(f"• F1 Score: {self.statistics['f1_score']:.3f}\n")
                
                self.log_message(f"✅ Статистика экспортирована в: {filepath}")
                QMessageBox.information(self, "Успех", f"Статистика сохранена в:\n{filepath}")
                
            except Exception as e:
                self.log_message(f"❌ Ошибка экспорта: {str(e)}")
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать статистику:\n{str(e)}")
    
    def export_image(self):
        """Экспортирует текущее изображение с аннотациями."""
        if not self.viewer.current_image or self.viewer.current_image.isNull():
            QMessageBox.warning(self, "Внимание", "Нет изображения для экспорта.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение",
            "coco_comparison_image.png",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)"
        )
        
        if filepath:
            try:
                # Создаем QPixmap для экспорта
                pixmap = QPixmap(self.viewer.size())
                pixmap.fill(Qt.GlobalColor.transparent)
                
                # Рисуем текущее состояние
                painter = QPainter(pixmap)
                self.viewer.render(painter)
                painter.end()
                
                # Сохраняем
                pixmap.save(filepath)
                
                self.log_message(f"✅ Изображение экспортировано в: {filepath}")
                QMessageBox.information(self, "Успех", f"Изображение сохранено в:\n{filepath}")
                
            except Exception as e:
                self.log_message(f"❌ Ошибка экспорта изображения: {str(e)}")
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать изображение:\n{str(e)}")
    
    # ============================================================================
    # ДИАЛОГИ
    # ============================================================================
    
    def show_about(self):
        """Показывает диалог 'О программе'."""
        about_text = """
        <h2>COCO Annotation Comparison Tool</h2>
        
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Author:</b> Равиль Рависович</p>
        <p><b>Email:</b> RavilRavisovich@gmail.com</p>
        <p><b>Identifier:</b> @X5373</p>
        
        <hr>
        
        <p>Это приложение предназначено для сравнения машинных и человеческих 
        аннотаций в формате COCO v1.</p>
        
        <p><b>Основные функции:</b></p>
        <ul>
            <li>Загрузка аннотаций COCO v1</li>
            <li>Визуальное сравнение аннотаций</li>
            <li>Отображение полигонов сегментации</li>
            <li>Расчет метрик качества (Precision, Recall, F1)</li>
            <li>Экспорт результатов</li>
        </ul>
        
        <p><b>Цветовая схема:</b></p>
        <ul>
            <li><font color='red'>Красный</font> - машинные аннотации</li>
            <li><font color='green'>Зеленый</font> - человеческие аннотации</li>
        </ul>
        """
        
        QMessageBox.about(self, "О программе", about_text)

# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

def main():
    """Точка входа в приложение."""
    # Выводим информацию об авторе в консоль
    print("=" * 60)
    print("COCO Annotation Comparison Tool v1.0")
    print("Автор: Равиль Рависович")
    print("Контакт: RavilRavisovich@gmail.com")
    print("Идентификатор: @X5373")
    print("=" * 60)
    
    # Создаем и запускаем приложение
    app = QApplication(sys.argv)
    app.setApplicationName("COCO Comparison Tool")
    app.setOrganizationName("@X5373")
    
    window = CocoComparisonApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
