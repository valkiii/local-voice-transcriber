# Just the fixed section
        analysis_controls.addLayout(model_layout)
        
        # Load transcription button
        self.load_button = QPushButton("📁 Load Transcription")
        self.load_button.clicked.connect(self.load_transcription_file)
        self.load_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF6B35, stop:1 #E55A2B);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
                margin: 2px 0px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF7B45, stop:1 #F56A3B);
            }
        """)
        analysis_controls.addWidget(self.load_button)