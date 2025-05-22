from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from pathlib import Path
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"##############Device: {device}##############")

class DarkModeStyle:
    """คลาสสำหรับกำหนดสีและสไตล์ Dark Mode"""
    # สีหลัก
    BG_COLOR = "#1e1e2e"           # สีพื้นหลักหลัก
    SECONDARY_BG = "#313244"       # สีพื้นหลังรอง
    TEXT_COLOR = "#cdd6f4"         # สีข้อความ
    ACCENT_COLOR = "#89b4fa"       # สีเน้น
    BUTTON_BG = "#45475a"          # สีพื้นหลังปุ่ม
    BUTTON_ACTIVE_BG = "#585b70"   # สีพื้นหลังปุ่มเมื่อกด
    
    # สีสำหรับปุ่มพิเศษ
    CAR_COLOR = "#a6e3a1"          # สีเขียว สำหรับปุ่ม Car
    NOTCAR_COLOR = "#f38ba8"       # สีแดง สำหรับปุ่ม Not Car
    SKIP_COLOR = "#f9e2af"         # สีเหลือง สำหรับปุ่ม Skip
    UNSELECT_COLOR = "#fab387"     # สีส้ม สำหรับปุ่ม Unselect
    
    # สีสำหรับข้อความ log
    LOG_BG = "#181825"             # สีพื้นหลังกล่อง log
    INFO_COLOR = "#89b4fa"         # สีข้อความข้อมูล
    SUCCESS_COLOR = "#a6e3a1"      # สีข้อความสำเร็จ
    WARNING_COLOR = "#f9e2af"      # สีข้อความเตือน
    ERROR_COLOR = "#f38ba8"        # สีข้อความผิดพลาด
    
    # ฟอนต์
    FONT_FAMILY = "Helvetica"      # ฟอนต์หลัก (มาตรฐานที่ทุก OS มี)
    FONT_SIZE = 10                 # ขนาดฟอนต์พื้นฐาน
    FONT_SIZE_LARGE = 12           # ขนาดฟอนต์ใหญ่
    
    # ระยะห่าง
    PADDING = 10                   # ระยะห่างมาตรฐาน

class YOLOImageSimilaritySorter:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Image Similarity Sorter")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.resizable(True, True)
        
        # Set Dark Mode theme
        self.style = DarkModeStyle()
        self.setup_theme()
        
        # Selected folder and target folders
        self.folder_path = ""
        self.target_folders = {}
        
        # Image file list
        self.image_files = []
        self.current_index = 0
        
        # YOLOv8 model
        self.model = None
        self.is_running = False
        
        # เพิ่มตัวแปรโหมด
        self.mode = "normal"  # "normal" หรือ "not_car_auto"
        
        # รูปภาพที่เหมือนกัน
        self.similar_images = []
        self.unselected_images = set()  # เก็บรูปที่กด "unselect"
        self.threshold = 0.8  # ค่าความคล้าย (0-1)
        
        # Target classes in ImageNet related to vehicles and people
        self.target_class_keywords = [
            # People
            'person', 'man', 'woman', 'child', 'boy', 'girl', 'human',
            # Bicycles
            'bicycle', 'bike', 'cycle', 'mountain bike',
            # Cars
            'car', 'automobile', 'cab', 'taxi', 'jeep', 'sport car', 'passenger car',
            # Motorcycles
            'motorcycle', 'motorbike', 'motor scooter', 'scooter',
            # Airplanes
            'airplane', 'aircraft', 'plane', 'airliner', 'warplane', 'jet',
            # Buses
            'bus', 'coach', 'minibus', 'trolleybus',
            # Trains
            'train', 'locomotive', 'railway', 'railroad car', 'passenger car',
            # Trucks
            'truck', 'pickup', 'tractor', 'trailer truck', 'delivery truck',
            # Boats
            'boat', 'ship', 'sailboat', 'speedboat', 'canoe', 'vessel'
        ]
        
        # Statistics
        self.stats = {
            'total': 0,
            'processed': 0,
            'car': 0,
            'not_car': 0,
            'auto_not_car': 0,
            'similar_moved': 0,
            'skipped': 0  # เพิ่มสถิติรูปที่ skip
        }
        
        # Setup UI
        self.setup_ui()
        self.setup_additional_styles()
        self.debug_check_methods()
        
    def setup_theme(self):
        """กำหนดธีม Dark Mode สำหรับ ttk"""
        self.root.configure(bg=self.style.BG_COLOR)
        
        # สร้างสไตล์สำหรับ ttk widgets
        style = ttk.Style()
        style.theme_use('clam')  # ใช้ธีม clam เป็นพื้นฐานเพราะปรับแต่งง่าย
        
        # กำหนดสีพื้นฐานของ ttk themes
        style.configure('.', 
                        background=self.style.BG_COLOR, 
                        foreground=self.style.TEXT_COLOR, 
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE))
        
        # Frame
        style.configure('TFrame', background=self.style.BG_COLOR)
        style.configure('Card.TFrame', background=self.style.SECONDARY_BG, 
                        relief=tk.RAISED, borderwidth=0)
        
        # LabelFrame
        style.configure('TLabelframe', background=self.style.BG_COLOR, 
                        foreground=self.style.TEXT_COLOR)
        style.configure('TLabelframe.Label', background=self.style.BG_COLOR, 
                        foreground=self.style.TEXT_COLOR)
        
        # Label
        style.configure('TLabel', background=self.style.BG_COLOR, 
                        foreground=self.style.TEXT_COLOR)
        style.configure('Header.TLabel', font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_LARGE, 'bold'))
        style.configure('Stats.TLabel', background=self.style.SECONDARY_BG, 
                        padding=self.style.PADDING)
        
        # Button
        style.configure('TButton', 
                        background=self.style.BUTTON_BG, 
                        foreground=self.style.TEXT_COLOR, 
                        borderwidth=0, 
                        focusthickness=3, 
                        focuscolor=self.style.ACCENT_COLOR)
        style.map('TButton', 
                  background=[('active', self.style.BUTTON_ACTIVE_BG)], 
                  foreground=[('active', self.style.TEXT_COLOR)])
                
        # Car Button
        style.configure('Car.TButton', 
                        background=self.style.CAR_COLOR, 
                        foreground='#000000', 
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_LARGE, 'bold'))
        style.map('Car.TButton', 
                  background=[('active', '#76de74')], 
                  foreground=[('active', '#000000')])
                
        # NotCar Button
        style.configure('NotCar.TButton', 
                        background=self.style.NOTCAR_COLOR, 
                        foreground='#000000', 
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_LARGE, 'bold'))
        style.map('NotCar.TButton', 
                  background=[('active', '#e64980')], 
                  foreground=[('active', '#000000')])
                
        # Skip Button
        style.configure('Skip.TButton', 
                        background=self.style.SKIP_COLOR, 
                        foreground='#000000', 
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE, 'bold'))
        style.map('Skip.TButton', 
                  background=[('active', '#f0c674')], 
                  foreground=[('active', '#000000')])
        
        # Unselect Button
        style.configure('Unselect.TButton', 
                        background=self.style.UNSELECT_COLOR, 
                        foreground='#000000', 
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE, 'bold'))
        style.map('Unselect.TButton', 
                  background=[('active', '#e6ad85')], 
                  foreground=[('active', '#000000')])
                
        # Accent Button
        style.configure('Accent.TButton', 
                        background=self.style.ACCENT_COLOR, 
                        foreground='#000000')
        style.map('Accent.TButton', 
                  background=[('active', '#65a7e6')], 
                  foreground=[('active', '#000000')])
                
        # Entry
        style.configure('TEntry', 
                        fieldbackground=self.style.SECONDARY_BG, 
                        background=self.style.SECONDARY_BG, 
                        foreground=self.style.TEXT_COLOR, 
                        borderwidth=0, 
                        darkcolor=self.style.SECONDARY_BG, 
                        lightcolor=self.style.SECONDARY_BG)
                        
        # Progressbar
        style.configure("TProgressbar", 
                        troughcolor=self.style.SECONDARY_BG, 
                        background=self.style.ACCENT_COLOR, 
                        borderwidth=0, 
                        thickness=20)
        
        # Scale
        style.configure("TScale", 
                        background=self.style.BG_COLOR,
                        troughcolor=self.style.SECONDARY_BG,
                        sliderrelief=tk.FLAT)

    def setup_ui(self):
        """Create main UI components"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=self.style.PADDING, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # =========== Top section: Header and folder selection ===========
        header_frame = ttk.Frame(main_frame, style='TFrame')
        header_frame.pack(fill=tk.X, pady=(0, self.style.PADDING))
        
        # App title
        header_label = ttk.Label(header_frame, 
                                text="YOLOv8 Image Similarity Sorter", 
                                style='Header.TLabel')
        header_label.pack(side=tk.LEFT, padx=(0, self.style.PADDING))
        
        # =========== Folder selection section ===========
        folder_frame = ttk.Frame(main_frame, padding=self.style.PADDING, style='Card.TFrame')
        folder_frame.pack(fill=tk.X, pady=(0, self.style.PADDING))
        
        # Folder label
        folder_label = ttk.Label(folder_frame, text="โฟลเดอร์รูปภาพ:", style='TLabel')
        folder_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Folder path display
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=50, style='TEntry')
        folder_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        # Browse folder button
        browse_button = ttk.Button(folder_frame, text="เลือกโฟลเดอร์", 
                                command=self.select_folder, style='Accent.TButton')
        browse_button.pack(side=tk.LEFT)
        
        # =========== Similarity Threshold section ===========
        threshold_frame = ttk.Frame(main_frame, padding=self.style.PADDING, style='Card.TFrame')
        threshold_frame.pack(fill=tk.X, pady=(0, self.style.PADDING))
        
        # Threshold label
        threshold_label = ttk.Label(threshold_frame, text="ค่าความคล้าย (Threshold):", style='TLabel')
        threshold_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Threshold slider
        self.threshold_var = tk.StringVar(value=f"{self.threshold:.2f}")
        
        # แสดงค่า threshold ปัจจุบัน
        threshold_value_label = ttk.Label(threshold_frame, textvariable=self.threshold_var, width=4)
        threshold_value_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # ปุ่มลด threshold
        decrease_btn = ttk.Button(threshold_frame, text="-", width=2, 
                                command=self.decrease_threshold)
        decrease_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # ปุ่มเพิ่ม threshold
        increase_btn = ttk.Button(threshold_frame, text="+", width=2, 
                                command=self.increase_threshold)
        increase_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Scale for threshold
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, 
                                        orient=tk.HORIZONTAL, length=200,
                                        command=self.update_threshold)
        self.threshold_scale.set(self.threshold)
        self.threshold_scale.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        # =========== Statistics section ===========
        stats_frame = ttk.Frame(main_frame, padding=self.style.PADDING, style='Card.TFrame')
        stats_frame.pack(fill=tk.X, pady=(0, self.style.PADDING))
        
        # Statistics header
        stats_header = ttk.Label(stats_frame, text="สถิติการคัดแยก", style='TLabel')
        stats_header.pack(anchor=tk.W, pady=(0, 5))
        
        # First statistics row
        stats_row1 = ttk.Frame(stats_frame, style='TFrame')
        stats_row1.pack(fill=tk.X)
        
        # Create variables for displaying each statistic
        self.total_var = tk.StringVar(value="รูปภาพทั้งหมด: 0")
        self.processed_var = tk.StringVar(value="ดำเนินการแล้ว: 0 (0%)")
        self.car_var = tk.StringVar(value="Car: 0")
        self.not_car_var = tk.StringVar(value="Not Car (ผู้ใช้): 0")
        self.auto_not_car_var = tk.StringVar(value="Not Car (อัตโนมัติ): 0")
        self.skipped_var = tk.StringVar(value="ข้าม: 0")  # เพิ่มตัวแปรสถิติ skip
        self.similar_moved_var = tk.StringVar(value="รูปที่คล้ายที่ย้ายแล้ว: 0")
        
        # Display statistics in a grid layout
        ttk.Label(stats_row1, textvariable=self.total_var, style='Stats.TLabel').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Label(stats_row1, textvariable=self.processed_var, style='Stats.TLabel').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # Second statistics row
        stats_row2 = ttk.Frame(stats_frame, style='TFrame')
        stats_row2.pack(fill=tk.X, pady=(1, 0))
        
        ttk.Label(stats_row2, textvariable=self.car_var, style='Stats.TLabel').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Label(stats_row2, textvariable=self.not_car_var, style='Stats.TLabel').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Label(stats_row2, textvariable=self.auto_not_car_var, style='Stats.TLabel').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # Third statistics row
        stats_row3 = ttk.Frame(stats_frame, style='TFrame')
        stats_row3.pack(fill=tk.X, pady=(1, 0))
        
        ttk.Label(stats_row3, textvariable=self.skipped_var, style='Stats.TLabel').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Label(stats_row3, textvariable=self.similar_moved_var, style='Stats.TLabel').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # =========== Progress bar ===========
        progress_frame = ttk.Frame(main_frame, style='TFrame')
        progress_frame.pack(fill=tk.X, pady=(0, self.style.PADDING))
        
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, 
                                        length=100, mode='determinate', style='TProgressbar')
        self.progress.pack(fill=tk.X)
        
        # =========== Control buttons ===========
        control_frame = ttk.Frame(main_frame, style='TFrame')
        control_frame.pack(fill=tk.X, pady=(0, self.style.PADDING))
        
        # ปุ่มเริ่มคัดแยกแบบปกติ
        self.start_button = ttk.Button(control_frame, text="เริ่มคัดแยก", 
                                    command=self.start_normal_sorting, state=tk.DISABLED, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # ปุ่มเริ่มโหมด Not Car Auto
        self.not_car_auto_button = ttk.Button(control_frame, text="Not Car Auto", 
                                            command=self.start_not_car_auto_sorting, state=tk.DISABLED, style='Car.TButton')
        self.not_car_auto_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # ปุ่มหยุด
        self.stop_button = ttk.Button(control_frame, text="หยุด", 
                                    command=self.stop_sorting, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # แสดงโหมดปัจจุบัน
        self.mode_var = tk.StringVar(value="โหมด: ปกติ")
        mode_label = ttk.Label(control_frame, textvariable=self.mode_var, style='TLabel')
        mode_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # =========== Log display section ===========
        log_frame = ttk.Frame(main_frame, style='Card.TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log label
        log_label = ttk.Label(log_frame, text="Log กิจกรรม", style='TLabel')
        log_label.pack(anchor=tk.W, padx=self.style.PADDING, pady=(self.style.PADDING, 0))
        
        # Text box for displaying logs
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, bg=self.style.LOG_BG, 
                            fg=self.style.TEXT_COLOR, insertbackground=self.style.TEXT_COLOR,
                            padx=5, pady=5, relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=self.style.PADDING, pady=self.style.PADDING)
        
        # Add scrollbar for log
        scrollbar = ttk.Scrollbar(self.log_text, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Define tags for different message types
        self.log_text.tag_configure("info", foreground=self.style.INFO_COLOR)
        self.log_text.tag_configure("success", foreground=self.style.SUCCESS_COLOR)
        self.log_text.tag_configure("warning", foreground=self.style.WARNING_COLOR)
        self.log_text.tag_configure("error", foreground=self.style.ERROR_COLOR)
        
        # Welcome message
        self.log("YOLOv8 Image Similarity Sorter started", "info")
        self.log("กรุณาเลือกโฟลเดอร์รูปภาพเพื่อเริ่มต้น")

    def start_normal_sorting(self):
        """เริ่มการคัดแยกแบบปกติ"""
        self.mode = "normal"
        self.mode_var.set("โหมด: ปกติ")
        self.log("เริ่มโหมดปกติ", "info")
        self.start_sorting()

    def start_not_car_auto_sorting(self):
        """เริ่มการคัดแยกแบบ Not Car Auto"""
        self.mode = "not_car_auto"
        self.mode_var.set("โหมด: Not Car Auto")
        self.log("เริ่มโหมด Not Car Auto", "success")
        self.start_sorting()
    
    def decrease_threshold(self):
        """ลดค่า threshold ลง 0.05"""
        new_threshold = max(0.1, self.threshold - 0.05)
        self.threshold = round(new_threshold, 2)
        self.threshold_var.set(f"{self.threshold:.2f}")
        self.threshold_scale.set(self.threshold)
        
    def increase_threshold(self):
        """เพิ่มค่า threshold ขึ้น 0.05"""
        new_threshold = min(1.0, self.threshold + 0.05)
        self.threshold = round(new_threshold, 2)
        self.threshold_var.set(f"{self.threshold:.2f}")
        self.threshold_scale.set(self.threshold)
        
    def update_threshold(self, value):
        """อัพเดทค่า threshold จาก scale"""
        self.threshold = float(value)
        self.threshold_var.set(f"{self.threshold:.2f}")
        
    def select_folder(self):
        """Open dialog to select a folder with images"""
        folder_path = filedialog.askdirectory(title="เลือกโฟลเดอร์ที่มีรูปภาพ")
        if folder_path:
            self.folder_path = folder_path
            self.folder_var.set(folder_path)
            # เปิดใช้งานปุ่มทั้งสอง
            self.start_button.config(state=tk.NORMAL)
            self.not_car_auto_button.config(state=tk.NORMAL)
            self.log(f"เลือกโฟลเดอร์: {folder_path}", "info")
            
    def log(self, message, level="normal"):
        """Add a message to the log area"""
        self.log_text.config(state=tk.NORMAL)
        
        # Add timestamp before message
        from datetime import datetime
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        self.log_text.insert(tk.END, timestamp, "info")
        
        # Add message with appropriate color based on level
        if level in ["info", "success", "warning", "error"]:
            self.log_text.insert(tk.END, message + "\n", level)
        else:
            self.log_text.insert(tk.END, message + "\n")
            
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()
        
    def update_stats(self):
        """Update the statistics text"""
        # Calculate progress percentage
        progress_percent = 0
        if self.stats['total'] > 0:
            progress_percent = (self.stats['processed'] / self.stats['total']) * 100
            
        # Update statistics variables
        self.total_var.set(f"รูปภาพทั้งหมด: {self.stats['total']}")
        self.processed_var.set(f"ดำเนินการแล้ว: {self.stats['processed']} ({progress_percent:.1f}%)")
        self.car_var.set(f"Car: {self.stats['car']}")
        self.not_car_var.set(f"Not Car (ผู้ใช้): {self.stats['not_car']}")
        self.auto_not_car_var.set(f"Not Car (อัตโนมัติ): {self.stats['auto_not_car']}")
        self.skipped_var.set(f"ข้าม: {self.stats['skipped']}")
        self.similar_moved_var.set(f"รูปที่คล้ายที่ย้ายแล้ว: {self.stats['similar_moved']}")
        
        # Update progress bar
        self.progress["value"] = progress_percent
        
    def start_sorting(self):
        """Start the image sorting process"""
        if not self.folder_path:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกโฟลเดอร์ที่มีรูปภาพก่อน")
            return
        
        if self.is_running:
            return
            
        self.is_running = True
            
        # Create target folders
        self.target_folders = {
            'car': os.path.join(self.folder_path, 'car'),
            'not_car': os.path.join(self.folder_path, 'not_car')
        }
        
        for folder in self.target_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Lock both start buttons to prevent multiple starts
        self.start_button.config(state=tk.DISABLED)
        self.not_car_auto_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
            
        # Start processing in a separate thread
        threading.Thread(target=self._sorting_thread, daemon=True).start()
    
    def _sorting_thread(self):
        """Function for working in a separate thread"""
        # Load YOLOv8 model
        try:
            self.log("กำลังโหลดโมเดล YOLOv8...", "info")
            self.model = YOLO('best-cls-v2.pt', ).to(0)
            self.log("โหลดโมเดลสำเร็จ", "success")
        except Exception as e:
            self.log(f"ข้อผิดพลาดในการโหลดโมเดล: {e}", "error")
            messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถโหลดโมเดลได้: {e}")
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.not_car_auto_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            return
            
        # Collect all image files in the folder
        self.collect_image_files()
        
        if not self.image_files:
            self.log("ไม่พบไฟล์ภาพในโฟลเดอร์", "warning")
            messagebox.showinfo("ข้อมูล", "ไม่พบไฟล์ภาพในโฟลเดอร์ที่เลือก")
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.not_car_auto_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            return
            
        # Update statistics - เพิ่ม 'skipped' ในนี้ด้วย
        self.stats = {
            'total': len(self.image_files),
            'processed': 0,
            'car': 0,
            'not_car': 0,
            'auto_not_car': 0,
            'similar_moved': 0,
            'skipped': 0  # เพิ่มบรรทัดนี้
        }
        self.root.after(0, self.update_stats)
        
        # Start processing images one by one
        self.current_index = 0
        self.process_next_image()
        
    def collect_image_files(self):
        """Collect all image files in the folder"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
        
        self.log("กำลังค้นหาไฟล์ภาพ...", "info")
        
        self.image_files = []
        for file in os.listdir(self.folder_path):
            # แก้ตรงนี้: ใช้ os.path.normpath เพื่อมาตรฐานเส้นทาง
            file_path = os.path.normpath(os.path.join(self.folder_path, file))
            
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in valid_extensions):
                if not any(target_folder in file_path for target_folder in self.target_folders.values()):
                    self.image_files.append(file_path)
                        
        self.log(f"พบไฟล์ภาพทั้งหมด {len(self.image_files)} ไฟล์", "success")
        
    def process_next_image(self):
        """Process the next image"""
        if not self.is_running or self.current_index >= len(self.image_files) or not self.image_files:
            self.finish_sorting()
            return
            
        img_path = self.image_files[self.current_index]
        filename = os.path.basename(img_path)
        
        self.log(f"กำลังประมวลผล ({self.current_index+1}/{len(self.image_files)}): {filename}")
        
        try:
            # Predict with model (classification)
            results = self.model(img_path)
            result = results[0]
            
            # Get the class with highest confidence
            class_id = result.probs.top1
            class_name = result.names[class_id]
            confidence = result.probs.top1conf.item()
            
            # Check if the predicted class is in our target classes
            is_target_class = any(keyword.lower() in class_name.lower() for keyword in self.target_class_keywords)
            
            self.log(f"  คลาสที่ทำนาย: {class_name}, ความมั่นใจ: {confidence:.4f}")
            
            if self.mode == "normal":
                # โหมดปกติ
                if is_target_class:
                    # If it's a target class, find similar images
                    self.log(f"  กำลังค้นหารูปที่คล้ายกัน...", "info")
                    
                    # Reset similar images and unselected set
                    self.similar_images = []
                    self.unselected_images = set()
                    
                    # Find similar images in a separate thread
                    self.root.after(0, lambda: self.find_similar_images(img_path, class_name, confidence))
                else:
                    # If not a target class, move to not_car folder automatically
                    target_path = os.path.join(self.target_folders['not_car'], filename)
                    shutil.move(img_path, target_path)
                    self.log(f"  ย้ายไปยังโฟลเดอร์ not_car โดยอัตโนมัติ", "info")
                    
                    # Update statistics
                    self.stats['processed'] += 1
                    self.stats['auto_not_car'] += 1
                    self.root.after(0, self.update_stats)
                    
                    # Process next image
                    self.current_index += 1
                    self.root.after(100, self.process_next_image)
                    
            elif self.mode == "not_car_auto":
                # โหมด Not Car Auto
                if is_target_class:
                    # ถ้าเจอรถ → Skip ไปเลย
                    self.log(f"  โหมด Not Car Auto: เจอรถ → Skip", "warning")
                    
                    # Update statistics
                    self.stats['processed'] += 1
                    self.stats['skipped'] += 1
                    self.root.after(0, self.update_stats)
                    
                    # Process next image
                    self.current_index += 1
                    self.root.after(100, self.process_next_image)
                else:
                    # ถ้าไม่เจอรถ → ย้ายไป not_car ทันที
                    target_path = os.path.join(self.target_folders['not_car'], filename)
                    shutil.move(img_path, target_path)
                    self.log(f"  โหมด Not Car Auto: ไม่เจอรถ → ย้ายไป not_car", "success")
                    
                    # Update statistics
                    self.stats['processed'] += 1
                    self.stats['auto_not_car'] += 1
                    self.root.after(0, self.update_stats)
                    
                    # Process next image
                    self.current_index += 1
                    self.root.after(100, self.process_next_image)
                    
        except Exception as e:
            self.log(f"ข้อผิดพลาดในการประมวลผลไฟล์ {filename}: {e}", "error")
            # Skip to next image
            self.current_index += 1
            self.root.after(100, self.process_next_image)
    
    def find_similar_images(self, img_path, class_name, confidence):
        """ค้นหารูปที่คล้ายกับรูปต้นแบบ"""
        try:
            # ใช้ thread แยกเพื่อไม่ให้ UI ค้าง
            threading.Thread(target=self._find_similar_images_thread, 
                           args=(img_path, class_name, confidence),
                           daemon=True).start()
        except Exception as e:
            self.log(f"ข้อผิดพลาดในการค้นหารูปที่คล้ายกัน: {e}", "error")
            # Skip to next image
            self.current_index += 1
            self.process_next_image()
    
    def _find_similar_images_thread(self, ref_img_path, class_name, confidence):
        """Thread สำหรับค้นหารูปที่คล้ายกัน (แบบ sliding window 10 รูปต่อรอบ)"""
        try:
            ref_image = cv2.imread(ref_img_path)
            if ref_image is None:
                self.root.after(0, lambda: self.log(f"ไม่สามารถอ่านรูปต้นแบบได้", "error"))
                self.current_index += 1
                self.root.after(0, self.process_next_image)
                return

            ref_image = cv2.resize(ref_image, (128, 128))
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

            # 1. Histogram
            ref_hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
            histSize = [8, 8]
            ranges = [0, 180, 0, 256]
            channels = [0, 1]
            ref_hist = cv2.calcHist([ref_hsv], channels, None, histSize, ranges)
            cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX)

            # 2. Average Hash
            small_img = cv2.resize(ref_gray, (8, 8))
            avg = small_img.mean()
            ref_avg_hash = small_img >= avg

            # 3. Difference Hash
            resize_img = cv2.resize(ref_gray, (9, 8))
            ref_d_hash = np.zeros((8, 8), dtype=np.bool_)
            for i in range(8):
                for j in range(8):
                    ref_d_hash[i, j] = resize_img[i, j] > resize_img[i, j + 1]

            # 4. ORB
            orb = cv2.ORB_create(nfeatures=1000)
            _, ref_features = orb.detectAndCompute(ref_gray, None)

            similar_images = []
            
            # หา index ของรูปต้นแบบ
            ref_index = -1
            for i, img_path in enumerate(self.image_files):
                if os.path.abspath(img_path) == os.path.abspath(ref_img_path):
                    ref_index = i
                    break
            
            if ref_index == -1:
                self.root.after(0, lambda: self.log(f"ไม่พบรูปต้นแบบในรายการ", "error"))
                self.current_index += 1
                self.root.after(0, self.process_next_image)
                return

            # เริ่มหารูปที่คล้ายกันแบบ sliding window
            window_size = 200
            start_index = ref_index + 1  # เริ่มจากรูปถัดไป
            
            while start_index < len(self.image_files):
                # กำหนดขอบเขตของ window ปัจจุบัน
                end_index = min(start_index + window_size, len(self.image_files))
                current_window = self.image_files[start_index:end_index]
                
                self.root.after(0, lambda s=start_index, e=end_index: 
                            self.log(f"  กำลังตรวจสอบรูปที่ {s+1}-{e} จากทั้งหมด {len(self.image_files)} รูป", "info"))
                
                # ตัวแปรเก็บว่าเจอรูปคล้ายใน window นี้หรือไม่
                found_similar_in_window = False
                
                # ตรวจสอบรูปใน window ปัจจุบัน
                for img_path in current_window:
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        img = cv2.resize(img, (128, 128))
                        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        img_hist = cv2.calcHist([img_hsv], channels, None, histSize, ranges)
                        cv2.normalize(img_hist, img_hist, 0, 1, cv2.NORM_MINMAX)
                        hist_similarity = cv2.compareHist(ref_hist, img_hist, cv2.HISTCMP_CORREL)
                        hist_similarity = max(0, hist_similarity)

                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        small_img = cv2.resize(img_gray, (8, 8))
                        avg = small_img.mean()
                        img_avg_hash = small_img >= avg
                        avg_hash_similarity = np.mean(ref_avg_hash == img_avg_hash)

                        resize_img = cv2.resize(img_gray, (9, 8))
                        img_d_hash = np.zeros((8, 8), dtype=np.bool_)
                        for i in range(8):
                            for j in range(8):
                                img_d_hash[i, j] = resize_img[i, j] > resize_img[i, j + 1]
                        d_hash_similarity = np.mean(ref_d_hash == img_d_hash)

                        # ORB feature matching
                        feature_similarity = 0
                        if ref_features is not None and ref_features.shape[0] > 0:
                            _, img_features = orb.detectAndCompute(img_gray, None)
                            if img_features is not None and img_features.shape[0] > 0:
                                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                                matches = bf.match(ref_features, img_features)
                                if len(matches) > 0:
                                    matches = sorted(matches, key=lambda x: x.distance)
                                    good_matches = [m for m in matches if m.distance < 50]
                                    if len(good_matches) > 0:
                                        feature_similarity = len(good_matches) / max(len(matches), 1)

                        if feature_similarity <= 0:
                            similarity_score = (
                                hist_similarity * 0.5 +
                                avg_hash_similarity * 0.2 +
                                d_hash_similarity * 0.3
                            )
                        else:
                            similarity_score = (
                                hist_similarity * 0.4 +
                                avg_hash_similarity * 0.2 +
                                d_hash_similarity * 0.3 +
                                feature_similarity * 0.1
                            )

                        if similarity_score >= self.threshold:
                            similar_images.append((img_path, similarity_score))
                            found_similar_in_window = True
                            
                            # แสดง log เมื่อเจอรูปคล้าย
                            filename = os.path.basename(img_path)
                            self.root.after(0, lambda f=filename, s=similarity_score: 
                                        self.log(f"    เจอรูปคล้าย: {f} (ความคล้าย: {s:.2%})", "success"))

                    except Exception as e:
                        self.root.after(0, lambda e=e, path=img_path: 
                                    self.log(f"ข้อผิดพลาดในการประมวลผลไฟล์ {os.path.basename(path)}: {e}", "error"))
                        continue

                # ตรวจสอบว่าเจอรูปคล้ายใน window นี้หรือไม่
                if found_similar_in_window:
                    self.root.after(0, lambda s=start_index, e=end_index: 
                                self.log(f"  เจอรูปคล้ายในรูปที่ {s+1}-{e}, ดำเนินการหาต่อ...", "info"))
                    # เลื่อน window ไปข้างหน้า
                    start_index = end_index
                else:
                    # ไม่เจอรูปคล้ายใน window นี้ หยุดการค้นหา
                    self.root.after(0, lambda s=start_index, e=end_index: 
                                self.log(f"  ไม่เจอรูปคล้ายในรูปที่ {s+1}-{e}, หยุดการค้นหา", "warning"))
                    break

            self.similar_images = similar_images
            self.root.after(0, lambda: self.log(f"พบรูปที่คล้ายกันทั้งหมด {len(similar_images)} รูป", "success"))
            self.root.after(0, lambda: self.show_decision_ui(ref_img_path, class_name, confidence, similar_images))

        except Exception as e:
            self.root.after(0, lambda e=e: self.log(f"ข้อผิดพลาดในการค้นหารูปที่คล้ายกัน: {e}", "error"))
            self.current_index += 1
            self.root.after(0, self.process_next_image)

    
    def show_decision_ui(self, ref_img_path, class_name, confidence, similar_images):
        """แสดง UI สำหรับการตัดสินใจและแสดงรูปที่คล้ายกัน"""
        # สร้างหน้าต่างใหม่
        decision_window = tk.Toplevel(self.root)
        decision_window.title("ตัดสินใจจัดหมวดหมู่รูปภาพ")
        decision_window.geometry("1200x800")
        decision_window.configure(bg=self.style.BG_COLOR)
        decision_window.protocol("WM_DELETE_WINDOW", lambda: self.on_decision_window_close(decision_window))
        
        # กำหนดให้แสดงเต็มจอ
        decision_window.state('zoomed')
        
        # ทำให้หน้าต่างอยู่ตรงกลางหน้าต่างหลัก
        decision_window.transient(self.root)
        decision_window.grab_set()
        
        # Frame หลัก
        main_frame = ttk.Frame(decision_window, style='TFrame', padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ==== ส่วนบน: รูปต้นแบบและปุ่มตัดสินใจ ====
        top_frame = ttk.Frame(main_frame, style='Card.TFrame', padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # แบ่งเป็น 2 ส่วน: รูปต้นแบบ และ ปุ่มตัดสินใจ
        top_left_frame = ttk.Frame(top_frame, style='TFrame')
        top_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        top_right_frame = ttk.Frame(top_frame, style='TFrame')
        top_right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # แสดงรูปต้นแบบ
        ttk.Label(top_left_frame, text="รูปต้นแบบ:", style='Header.TLabel', 
                 background=self.style.SECONDARY_BG).pack(anchor=tk.W, pady=(0, 5))
        
        try:
            # อ่านรูปและปรับขนาด
            ref_img = Image.open(ref_img_path)
            ref_img.thumbnail((400, 300))
            ref_photo = ImageTk.PhotoImage(ref_img)
            
            # เก็บรูปไว้ใน attributes ของหน้าต่าง
            decision_window.ref_photo = ref_photo
            
            # แสดงรูป
            ref_img_label = ttk.Label(top_left_frame, image=ref_photo, background=self.style.SECONDARY_BG)
            ref_img_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # แสดงข้อมูลรูป
            info_frame = ttk.Frame(top_left_frame, style='TFrame')
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            filename = os.path.basename(ref_img_path)
            info_text = (
                f"ไฟล์: {filename}\n"
                f"คลาสที่ทำนาย: {class_name}\n"
                f"ความมั่นใจ: {confidence:.4f}\n"
                f"พบรูปที่คล้ายกัน: {len(similar_images)} รูป"
            )
            
            ttk.Label(info_frame, text=info_text, background=self.style.SECONDARY_BG,
                     justify=tk.LEFT).pack(anchor=tk.W)
            
            # ปุ่มตัดสินใจ
            buttons_frame = ttk.Frame(top_right_frame, style='TFrame')
            buttons_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(buttons_frame, text="ตัดสินใจ:", style='Header.TLabel',
                    background=self.style.SECONDARY_BG).pack(anchor=tk.CENTER, pady=(0, 10))
            
            # ปุ่ม Car
            car_btn = ttk.Button(
                buttons_frame, 
                text="CAR (c)", 
                command=lambda: self.on_category_decision(decision_window, ref_img_path, 'car'),
                style='Car.TButton',
                padding=(20, 10)
            )
            car_btn.pack(fill=tk.X, pady=5)
            
            # ปุ่ม Not Car
            not_car_btn = ttk.Button(
                buttons_frame, 
                text="NOT CAR (n)", 
                command=lambda: self.on_category_decision(decision_window, ref_img_path, 'not_car'),
                style='NotCar.TButton',
                padding=(20, 10)
            )
            not_car_btn.pack(fill=tk.X, pady=5)
            
            # ปุ่ม Skip
            skip_btn = ttk.Button(
                buttons_frame, 
                text="ข้าม (s)", 
                command=lambda: self.on_category_decision(decision_window, ref_img_path, 'skip'),
                style='Skip.TButton',
                padding=(10, 5)
            )
            skip_btn.pack(fill=tk.X, pady=5)
            
            # ==== ส่วนล่าง: รูปที่คล้ายกัน (scrollable) ====
            bottom_frame = ttk.Frame(main_frame, style='Card.TFrame', padding=10)
            bottom_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(bottom_frame, text="รูปที่คล้ายกัน:", style='Header.TLabel', 
                     background=self.style.SECONDARY_BG).pack(anchor=tk.W, pady=(0, 5))
            
            # สร้าง scrollable frame สำหรับแสดงรูปที่คล้ายกัน
            canvas = tk.Canvas(bottom_frame, bg=self.style.SECONDARY_BG, 
                             highlightthickness=0)
            scrollbar = ttk.Scrollbar(bottom_frame, orient="vertical", command=canvas.yview)
            
            scrollable_frame = ttk.Frame(canvas, style='TFrame')
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # เพิ่ม mouse wheel scrolling
            canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
            
            # แสดงรูปที่คล้ายกัน
            self.display_similar_images(scrollable_frame, similar_images)
            
            # เพิ่มการตรวจจับปุ่มกด
            decision_window.bind("<KeyPress>", lambda event: self.on_key_press_decision(event, decision_window, ref_img_path))
            
        except Exception as e:
            self.log(f"ข้อผิดพลาดในการแสดง UI: {e}", "error")
            decision_window.destroy()
            # Skip to next image
            self.current_index += 1
            self.process_next_image()
    
    def display_similar_images(self, parent_frame, similar_images):
        """แสดงรูปที่คล้ายกันใน scrollable frame (แก้ไข event binding)"""
        print(f"📷 แสดงรูป {len(similar_images)} รูป")
        
        if not similar_images:
            ttk.Label(parent_frame, text="ไม่พบรูปที่คล้ายกัน", 
                    background=self.style.SECONDARY_BG).pack(pady=10)
            return
        
        # กำหนดจำนวนคอลัมน์
        max_cols = 4
        
        # ตัวแปรสำหรับ drag functionality
        parent_frame.is_dragging = False
        parent_frame.drag_start_widget = None
        
        # สร้าง grid layout
        for i, (img_path, similarity) in enumerate(similar_images):
            row = i // max_cols
            col = i % max_cols
            
            try:
                print(f"🔗 สร้างรูป {i+1}: {os.path.basename(img_path)}")
                
                # สร้าง frame สำหรับแต่ละรูป
                img_frame = ttk.Frame(parent_frame, style='Card.TFrame', padding=5)
                img_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                
                # ให้ frame ขยายเมื่อหน้าต่างถูกปรับขนาด
                img_frame.columnconfigure(0, weight=1)
                img_frame.rowconfigure(0, weight=1)
                
                # อ่านรูปและปรับขนาด
                img = Image.open(img_path)
                img.thumbnail((250, 200))
                photo = ImageTk.PhotoImage(img)
                
                # เก็บ reference ของรูปไว้ใน dictionary
                if not hasattr(parent_frame, 'photo_references'):
                    parent_frame.photo_references = {}
                parent_frame.photo_references[img_path] = photo
                
                # แสดงรูป (พร้อม click event)
                img_label = ttk.Label(img_frame, image=photo, background=self.style.SECONDARY_BG)
                img_label.pack(pady=(0, 5))
                
                # แสดงข้อมูล
                filename = os.path.basename(img_path)
                info_text = f"{filename}\nความคล้าย: {similarity:.2%}"
                info_label = ttk.Label(img_frame, text=info_text, background=self.style.SECONDARY_BG,
                                    justify=tk.CENTER, wraplength=200)
                info_label.pack(pady=(0, 5))
                
                # ปุ่ม Unselect (เหมือนเดิม)
                unselect_btn = ttk.Button(
                    img_frame,
                    text="Unselect",
                    command=lambda path=img_path, frame=img_frame: self.toggle_image_selection(path, frame),
                    style='Unselect.TButton'
                )
                unselect_btn.pack(pady=(0, 5))
                
                # เก็บข้อมูลไว้ใน frame
                img_frame.unselect_btn = unselect_btn
                img_frame.img_path = img_path
                img_frame.img_label = img_label
                img_frame.info_label = info_label
                
                print(f"   📎 Binding events สำหรับรูป {i+1}...")
                
                # 🔧 แก้ไข: แยก event binding ออกจากกัน และใช้ลำดับที่ถูกต้อง
                
                # 1. ผูก Click Event ก่อน (สำคัญที่สุด)
                def make_click_handler(path, frame):
                    def handler(event):
                        print(f"🖱️ CLICK EVENT: {os.path.basename(path)}")
                        self.on_image_click(event, path, frame)
                    return handler
                
                click_handler = make_click_handler(img_path, img_frame)
                
                # ผูก click กับทั้งรูปและข้อมูล
                img_label.bind("<Button-1>", click_handler)
                info_label.bind("<Button-1>", click_handler)
                
                print(f"   ✅ Click events bound สำหรับรูป {i+1}")
                
                # 2. ผูก Drag Events (อย่าให้ชนกับ Click)
                def make_drag_start_handler(path, frame, parent):
                    def handler(event):
                        print(f"🎯 DRAG START: {os.path.basename(path)}")
                        self.on_drag_start(event, path, frame, parent)
                    return handler
                
                def make_drag_motion_handler(parent):
                    def handler(event):
                        self.on_drag_motion(event, parent)
                    return handler
                
                def make_drag_end_handler(parent):
                    def handler(event):
                        print(f"🔚 DRAG END")
                        self.on_drag_end(event, parent)
                    return handler
                
                drag_start_handler = make_drag_start_handler(img_path, img_frame, parent_frame)
                drag_motion_handler = make_drag_motion_handler(parent_frame)
                drag_end_handler = make_drag_end_handler(parent_frame)
                
                # ผูก drag events เฉพาะกับรูป (ไม่ใช่ข้อมูล)
                img_label.bind("<ButtonPress-1>", drag_start_handler, add="+")  # add="+" เพื่อไม่ override click
                img_label.bind("<B1-Motion>", drag_motion_handler)
                img_label.bind("<ButtonRelease-1>", drag_end_handler)
                
                print(f"   ✅ Drag events bound สำหรับรูป {i+1}")
                
                # 3. ผูก Hover Events
                def make_hover_enter_handler(frame):
                    def handler(event):
                        self.on_image_hover_enter(event, frame)
                    return handler
                
                def make_hover_leave_handler(frame):
                    def handler(event):
                        self.on_image_hover_leave(event, frame)
                    return handler
                
                hover_enter_handler = make_hover_enter_handler(img_frame)
                hover_leave_handler = make_hover_leave_handler(img_frame)
                
                img_label.bind("<Enter>", hover_enter_handler)
                img_label.bind("<Leave>", hover_leave_handler)
                
                print(f"   ✅ Hover events bound สำหรับรูป {i+1}")
                
                # 4. อัปเดตสถานะ visual เริ่มต้น
                self.update_image_visual_state(img_frame)
                
                print(f"   ✅ รูป {i+1} เสร็จสมบูรณ์: {os.path.basename(img_path)}")
                
            except Exception as e:
                print(f"❌ Error displaying image {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # กำหนดน้ำหนักของทุก column เท่ากัน
        for col in range(max_cols):
            parent_frame.columnconfigure(col, weight=1)
        
        print(f"✅ display_similar_images เสร็จสิ้น - สร้าง {len(similar_images)} รูป")

    # 2. แก้ไข on_image_click ให้ robust ขึ้น
    def on_image_click(self, event, img_path, img_frame):
        """จัดการเมื่อคลิกที่รูป (แก้ไขให้ robust)"""
        print(f"🖱️ on_image_click ถูกเรียก: {os.path.basename(img_path)}")
        print(f"   Event widget: {event.widget}")
        print(f"   Event type: {event.type}")
        
        try:
            # หา parent_frame จาก img_frame
            parent_frame = img_frame.master
            print(f"   Parent frame: {parent_frame}")
            
            # ตรวจสอบสถานะ drag (ป้องกันการ trigger เมื่อ drag)
            if hasattr(parent_frame, 'is_dragging') and parent_frame.is_dragging:
                print("   ⚠️ ข้ามการคลิกขณะลาก")
                return
                
            print("   🎯 เรียก toggle_image_selection")
            self.toggle_image_selection(img_path, img_frame)
            
        except Exception as e:
            print(f"   ❌ Error ใน on_image_click: {e}")
            import traceback
            traceback.print_exc()

    # 3. แก้ไข on_drag_start เพื่อไม่ให้ชนกับ click
    def on_drag_start(self, event, img_path, img_frame, parent_frame):
        """เริ่มต้นการ drag (แก้ไขเพื่อไม่ชนกับ click)"""
        print(f"🎯 on_drag_start: {os.path.basename(img_path)}")
        
        # ตั้งค่าเริ่มต้น (ยังไม่ถือว่าเป็น drag)
        parent_frame.is_dragging = False
        parent_frame.drag_start_widget = event.widget
        parent_frame.drag_start_x = event.x_root
        parent_frame.drag_start_y = event.y_root
        parent_frame.drag_start_time = event.time
        
        print(f"   ✅ Drag initialized")

    # 4. แก้ไข on_drag_motion
    def on_drag_motion(self, event, parent_frame):
        """จัดการการเคลื่อนไหวขณะ drag (แก้ไข threshold)"""
        if not hasattr(parent_frame, 'drag_start_widget') or parent_frame.drag_start_widget is None:
            return
            
        # ตรวจสอบว่าเมาส์เคลื่อนไหวพอสมควรแล้วหรือยัง
        if not parent_frame.is_dragging:
            dx = abs(event.x_root - parent_frame.drag_start_x)
            dy = abs(event.y_root - parent_frame.drag_start_y)
            # เพิ่ม threshold และเวลา เพื่อแยกจาก click
            if (dx > 10 or dy > 10) and (event.time - parent_frame.drag_start_time) > 100:
                parent_frame.is_dragging = True
                print("🔄 เริ่มโหมดลาก")

    # 5. เพิ่มฟังก์ชันทดสอบ
    def test_click_binding(self):
        """ทดสอบการ bind click events"""
        print("🧪 ทดสอบ click binding...")
        
        # สร้าง test window
        test_window = tk.Toplevel(self.root)
        test_window.title("Test Click Binding")
        test_window.geometry("400x300")
        test_window.configure(bg=self.style.BG_COLOR)
        
        def test_click(event):
            print(f"✅ Test click ทำงาน! Widget: {event.widget}")
            messagebox.showinfo("Test", f"Test click ทำงาน!\nWidget: {event.widget}")
        
        # สร้าง test widgets
        test_label1 = tk.Label(test_window, text="คลิกที่นี่ (Label)", 
                            bg="lightblue", width=30, height=3)
        test_label1.pack(pady=10)
        test_label1.bind("<Button-1>", test_click)
        
        test_label2 = ttk.Label(test_window, text="คลิกที่นี่ (TTK Label)")
        test_label2.pack(pady=10)
        test_label2.bind("<Button-1>", test_click)
        
        test_button = ttk.Button(test_window, text="ปิดทดสอบ", 
                                command=test_window.destroy)
        test_button.pack(pady=10)
        
        print("🎯 ลองคลิกที่ test widgets")

    def on_drag_end(self, event, parent_frame):
        """จบการ drag"""
        if not hasattr(parent_frame, 'is_dragging'):
            print("⚠️ parent_frame ไม่มี is_dragging")
            return
            
        if parent_frame.is_dragging:
            print("🔚 จบการลาก")
            try:
                # หา widget ที่ปล่อยเมาส์
                widget_under_mouse = event.widget.winfo_containing(event.x_root, event.y_root)
                if widget_under_mouse:
                    img_frame = self.find_img_frame_from_widget(widget_under_mouse)
                    if img_frame and hasattr(img_frame, 'img_path'):
                        # Toggle สถานะของรูปที่ปล่อยเมาส์
                        print(f"🎯 ลากไปที่: {os.path.basename(img_frame.img_path)}")
                        self.toggle_image_selection(img_frame.img_path, img_frame)
                    else:
                        print("⚠️ ไม่พบ img_frame ปลายทาง")
                else:
                    print("⚠️ ไม่พบ widget ใต้เมาส์")
            except Exception as e:
                print(f"❌ Error ใน on_drag_end: {e}")
        
        # รีเซ็ตสถานะ drag
        parent_frame.is_dragging = False
        parent_frame.drag_start_widget = None

    def find_img_frame_from_widget(self, widget):
        """หา img_frame จาก widget ใดๆ ภายใน"""
        current = widget
        while current:
            if hasattr(current, 'img_path'):  # นี่คือ img_frame
                return current
            try:
                current = current.master
            except:
                break
        return None

    def on_image_hover_enter(self, event, img_frame):
        """เมื่อเมาส์เข้าใกล้รูป"""
        if hasattr(img_frame, 'img_label'):
            img_frame.configure(style='Hover.TFrame')

    def on_image_hover_leave(self, event, img_frame):
        """เมื่อเมาส์ออกจากรูป"""
        # กลับไปใช้สีตามสถานะปกติ
        self.update_image_visual_state(img_frame)

    def toggle_image_selection(self, img_path, img_frame):
        """สลับการเลือก/ไม่เลือกรูปภาพ (เพิ่ม debug และ visual update)"""
        print(f"🔄 toggle_image_selection: {os.path.basename(img_path)}")
        
        try:
            if img_path in self.unselected_images:
                # ถ้าเคยอยู่ใน unselected ให้นำออก (เป็น selected)
                self.unselected_images.remove(img_path)
                img_frame.unselect_btn.config(text="Unselect", style='Unselect.TButton')
                print(f"🟢 เปลี่ยนเป็น Selected")
            else:
                # ถ้าไม่เคยอยู่ใน unselected ให้เพิ่มเข้าไป (เป็น unselected)
                self.unselected_images.add(img_path)
                img_frame.unselect_btn.config(text="Selected", style='TButton')
                print(f"🔴 เปลี่ยนเป็น Unselected")
            
            # อัปเดต visual state
            if hasattr(self, 'update_image_visual_state'):
                print(f"🎨 เรียก update_image_visual_state")
                self.update_image_visual_state(img_frame)
            else:
                print(f"❌ update_image_visual_state ไม่พบ!")
                
        except Exception as e:
            print(f"❌ Error ใน toggle_image_selection: {e}")

    def update_image_visual_state(self, img_frame):
        """อัปเดตสีพื้นหลังตามสถานะของรูป"""
        if not hasattr(img_frame, 'img_path'):
            print("⚠️ img_frame ไม่มี img_path")
            return
            
        try:
            if img_frame.img_path in self.unselected_images:
                # Unselected - ใช้สีแดงอ่อน
                img_frame.configure(style='Unselected.TFrame')
                print(f"🔴 Visual: Unselected - {os.path.basename(img_frame.img_path)}")
            else:
                # Selected - ใช้สีเขียวอ่อน
                img_frame.configure(style='Selected.TFrame')
                print(f"🟢 Visual: Selected - {os.path.basename(img_frame.img_path)}")
        except Exception as e:
            print(f"❌ Error ใน update_image_visual_state: {e}")

    def setup_additional_styles(self):
        """เพิ่มสไตล์สำหรับ visual states"""
        print("🎨 กำลังตั้งค่าสไตล์เพิ่มเติม...")
        
        try:
            style = ttk.Style()
            
            # Selected state
            style.configure('Selected.TFrame', 
                            background='#2d4a2d',  # เขียวเข้ม
                            relief=tk.RAISED, 
                            borderwidth=2)
            print("   ✅ Selected.TFrame")
            
            # Unselected state  
            style.configure('Unselected.TFrame', 
                            background='#4a2d2d',  # แดงเข้ม
                            relief=tk.RAISED, 
                            borderwidth=2)
            print("   ✅ Unselected.TFrame")
            
            # Hover state
            style.configure('Hover.TFrame', 
                            background='#4a4a2d',  # เหลืองเข้ม
                            relief=tk.RAISED, 
                            borderwidth=2)
            print("   ✅ Hover.TFrame")
            
            print("✅ Setup additional styles เสร็จสิ้น")
            
        except Exception as e:
            print(f"❌ Error ใน setup_additional_styles: {e}")

    def debug_check_methods(self):
        """ตรวจสอบว่ามีฟังก์ชันครบหรือไม่"""
        print("🔍 ตรวจสอบฟังก์ชัน:")
        required_methods = [
            'on_image_click',
            'on_drag_start', 
            'on_drag_motion',
            'on_drag_end',
            'update_image_visual_state',
            'setup_additional_styles',
            'toggle_image_selection'
        ]
        
        for method in required_methods:
            if hasattr(self, method):
                print(f"   ✅ {method}")
            else:
                print(f"   ❌ {method} - ไม่พบ!")
    
    def on_key_press_decision(self, event, window, ref_img_path):
        """จัดการเมื่อมีการกดปุ่มในหน้าต่างตัดสินใจ"""
        key = event.char.lower()
        if key == 'c':
            self.on_category_decision(window, ref_img_path, 'car')
        elif key == 'n':
            self.on_category_decision(window, ref_img_path, 'not_car')
        elif key == 's':
            self.on_category_decision(window, ref_img_path, 'skip')
    
    def on_category_decision(self, window, ref_img_path, category):
        """ดำเนินการย้ายรูปตามการตัดสินใจ (แบบใหม่: Unselected ไปตรงข้าม)"""
        if category == 'skip':
            window.destroy()
            # Skip to next image
            self.current_index += 1
            self.process_next_image()
            return
            
        # ย้ายรูปต้นแบบ
        ref_filename = os.path.basename(ref_img_path)
        ref_target_path = os.path.join(self.target_folders[category], ref_filename)
        
        try:
            shutil.move(ref_img_path, ref_target_path)
            self.log(f"ย้ายรูปต้นแบบไปยังโฟลเดอร์ {category}: {ref_filename}", "success")
            
            # อัปเดตสถิติตามหมวดหมู่ที่เลือก
            if category == 'car':
                self.stats['car'] += 1
            else:  # not_car
                self.stats['not_car'] += 1
                
            self.stats['processed'] += 1
            
            # ย้ายรูปที่คล้ายกัน (แบบใหม่: แยกตาม selected/unselected)
            moved_selected_count = 0
            moved_unselected_count = 0
            error_count = 0
            
            # กำหนดโฟลเดอร์ตรงข้าม
            opposite_category = 'not_car' if category == 'car' else 'car'
            
            for img_path, _ in self.similar_images:
                # ข้ามตัวเอง
                if os.path.abspath(img_path) == os.path.abspath(ref_img_path):
                    continue
                    
                try:
                    filename = os.path.basename(img_path)
                    
                    # ตรวจสอบว่ารูปนี้ถูก unselect หรือไม่
                    if img_path in self.unselected_images:
                        # รูป Unselected → ไปโฟลเดอร์ตรงข้าม
                        target_path = os.path.join(self.target_folders[opposite_category], filename)
                        
                        # ตรวจสอบชื่อซ้ำ
                        if os.path.exists(target_path):
                            name, ext = os.path.splitext(filename)
                            target_path = os.path.join(self.target_folders[opposite_category], 
                                                    f"{name}_unselected_{moved_unselected_count}{ext}")
                            
                        shutil.move(img_path, target_path)
                        moved_unselected_count += 1
                        
                        self.log(f"  รูป Unselected: {filename} → {opposite_category}", "warning")
                        
                    else:
                        # รูป Selected → ไปโฟลเดอร์ตามที่เลือก
                        target_path = os.path.join(self.target_folders[category], filename)
                        
                        # ตรวจสอบชื่อซ้ำ
                        if os.path.exists(target_path):
                            name, ext = os.path.splitext(filename)
                            target_path = os.path.join(self.target_folders[category], 
                                                    f"{name}_selected_{moved_selected_count}{ext}")
                            
                        shutil.move(img_path, target_path)
                        moved_selected_count += 1
                        
                        self.log(f"  รูป Selected: {filename} → {category}", "success")
                        
                except Exception as e:
                    self.log(f"ข้อผิดพลาดในการย้ายรูป {filename}: {e}", "error")
                    error_count += 1
            
            # อัปเดตสถิติรูปที่คล้ายกัน
            total_moved = moved_selected_count + moved_unselected_count
            self.stats['similar_moved'] += total_moved
            
            # แสดงสรุปผลลัพธ์
            summary_msg = (
                f"สรุปการย้ายรูป:\n"
                f"  - รูป Selected → {category}: {moved_selected_count} รูป\n"
                f"  - รูป Unselected → {opposite_category}: {moved_unselected_count} รูป\n"
                f"  - ข้อผิดพลาด: {error_count} รูป"
            )
            self.log(summary_msg, "info")
            
        except Exception as e:
            self.log(f"ข้อผิดพลาดในการย้ายรูปต้นแบบ: {e}", "error")
        
        # อัปเดตสถิติ
        self.root.after(0, self.update_stats)
        
        # ปิดหน้าต่าง
        window.destroy()
        
        # ไปยังรูปถัดไป
        self.current_index += 1
        self.process_next_image()
    
    def on_decision_window_close(self, window):
        """จัดการเมื่อปิดหน้าต่างตัดสินใจ (เทียบเท่ากับการกด Skip)"""
        window.destroy()
        # Skip to next image
        self.current_index += 1
        self.process_next_image()
    
    def stop_sorting(self):
        """หยุดกระบวนการคัดแยกรูปภาพ"""
        if not self.is_running:
            return
            
        # ตั้งค่าตัวแปรเพื่อหยุดการทำงาน
        self.is_running = False
        self.log("กำลังหยุดการทำงาน...", "warning")
        
        # อัปเดตสถานะปุ่ม
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.not_car_auto_button.config(state=tk.NORMAL)
        
    def finish_sorting(self):
        """เสร็จสิ้นกระบวนการคัดแยกรูปภาพ"""
        self.is_running = False
        
        # อัปเดตสถานะปุ่ม
        self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.not_car_auto_button.config(state=tk.NORMAL))
        
        # แสดงสรุปผล
        summary = (
            f"\nสรุปผลการทำงาน:\n"
            f"โหมด: {self.mode_var.get()}\n"
            f"จำนวนภาพทั้งหมด: {self.stats['total']} ไฟล์\n"
            f"จำนวนภาพที่ดำเนินการ: {self.stats['processed']} ไฟล์\n"
            f"จำนวนภาพที่จัดเป็น 'car': {self.stats['car']} ไฟล์\n"
            f"จำนวนภาพที่จัดเป็น 'not car' (จากผู้ใช้): {self.stats['not_car']} ไฟล์\n"
            f"จำนวนภาพที่จัดเป็น 'not car' (อัตโนมัติ): {self.stats['auto_not_car']} ไฟล์\n"
            f"จำนวนภาพที่ข้าม: {self.stats['skipped']} ไฟล์\n"
            f"จำนวนภาพที่คล้ายกันที่ถูกย้าย: {self.stats['similar_moved']} ไฟล์\n"
        )
        
        self.log(summary, "success")
        
        # แสดงข้อความสรุปในกล่องข้อความแจ้งเตือน
        self.root.after(100, lambda: messagebox.showinfo("สรุปผลการทำงาน", summary))
        
    def resize_image_for_display(self, image, max_width=800, max_height=600):
        """ปรับขนาดภาพให้พอดีกับหน้าจอ แต่ยังคงอัตราส่วนเดิม"""
        height, width = image.shape[:2]
        
        # ถ้าภาพมีขนาดเล็กกว่าที่กำหนดแล้ว ไม่ต้องปรับขนาด
        if width <= max_width and height <= max_height:
            return image
        
        # คำนวณอัตราส่วนการปรับขนาด
        ratio_width = max_width / width
        ratio_height = max_height / height
        ratio = min(ratio_width, ratio_height)
        
        # คำนวณขนาดใหม่
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # ปรับขนาดภาพ
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized_image

def main():
    # ตรวจสอบ dependencies ที่จำเป็น
    try:
        # นำเข้า PIL สำหรับการแสดงรูปภาพ
        global Image, ImageTk
        from PIL import Image, ImageTk
    except ImportError:
        print("กรุณาติดตั้ง Pillow ด้วยคำสั่ง: pip install pillow")
        import tkinter.messagebox as msg
        msg.showerror("ข้อผิดพลาด", "กรุณาติดตั้ง Pillow ด้วยคำสั่ง: pip install pillow")
        return
    
    try:
        # ตรวจสอบ OpenCV
        import cv2
    except ImportError:
        print("กรุณาติดตั้ง OpenCV ด้วยคำสั่ง: pip install opencv-python")
        import tkinter.messagebox as msg
        msg.showerror("ข้อผิดพลาด", "กรุณาติดตั้ง OpenCV ด้วยคำสั่ง: pip install opencv-python")
        return
    
    try:
        # ตรวจสอบ YOLO
        import ultralytics
    except ImportError:
        print("กรุณาติดตั้ง Ultralytics ด้วยคำสั่ง: pip install ultralytics")
        import tkinter.messagebox as msg
        msg.showerror("ข้อผิดพลาด", "กรุณาติดตั้ง Ultralytics ด้วยคำสั่ง: pip install ultralytics")
        return
    
    # สร้างสไตล์สำหรับปุ่มกด (ต้องกำหนดก่อนสร้าง Tk)
    try:
        # ตั้งค่าธีมหลัก
        import sv_ttk
        
        # สร้างหน้าต่างหลัก
        root = tk.Tk()
        
        # ลองใช้ธีม Sun Valley ถ้ามี
        try:
            sv_ttk.set_theme("dark")
            # แสดงข้อความว่าใช้ธีม Sun Valley
            print("Using Sun Valley theme (dark mode)")
        except Exception:
            # ถ้าไม่สามารถใช้ธีม Sun Valley ได้ ให้ใช้ธีมปกติ
            print("Sun Valley theme not available, using custom dark theme")
    except ImportError:
        # ถ้าไม่มี sv_ttk ให้ใช้ธีมปกติ
        root = tk.Tk()
        print("Sun Valley theme not installed, using custom dark theme")
    
    # กำหนดไอคอนแอปพลิเคชัน (ถ้ามี)
    try:
        # สำหรับ Windows
        root.iconbitmap("icon.ico")
    except:
        # ถ้าไม่มีไอคอน หรือไม่ใช่ Windows ให้ข้าม
        pass
    
    # เริ่มแอปพลิเคชัน
    app = YOLOImageSimilaritySorter(root)
    root.mainloop()

if __name__ == "__main__":
    main()