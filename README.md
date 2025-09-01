

````markdown
# Fingerprint Matcher 

A Python application for robust **fingerprint verification**, complete with visual reports, skeleton SVG exports, and an intuitive GUI.



## 🚀 Features

- **Automatic Preprocessing**: Auto-cropping, denoising, and enhancement of fingerprint images.  
- **Skeletonization**: Converts fingerprints to skeletons for accurate feature extraction.  
- **Feature Matching**: Uses ORB features and RANSAC for reliable fingerprint comparison.  
- **Bubble Visualization**: Displays matched points as bubbles instead of cluttered lines; mismatches are clearly marked.  
- **PDF Reports**: Generate detailed reports including match previews, thumbnails, skeletons, and statistics.  
- **SVG Exports**: Save skeletons as vector graphics for further analysis.  
- **GUI**: Clickable thumbnails, threaded matching, and progress indicators for smooth interaction.

---

## 🛠️ Built With

- Python 3.x  
- [OpenCV](https://opencv.org/)  
- [Pillow (PIL)](https://pillow.readthedocs.io/)  
- [scikit-image](https://scikit-image.org/)  
- [Tkinter](https://docs.python.org/3/library/tkinter.html)  
- [ReportLab](https://www.reportlab.com/)  
- [svgwrite](https://pypi.org/project/svgwrite/)  
- Matplotlib, NumPy  

---

## 💻 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/mubashirabbas/fingerprint-matcher-v7.git
cd fingerprint-matcher-v7
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*Dependencies include:* `opencv-python`, `numpy`, `Pillow`, `scikit-image`, `reportlab`, `svgwrite`, `matplotlib`

### 3. Run the application

```bash
python fingerprint_app_v7.py
```

---

## 🖼️ How to Use

1. Click **Load Query** to select the fingerprint to verify.
2. Click **Load Target** to select the fingerprint to compare.
3. Click **Match** to perform fingerprint verification.
4. Preview matched points on the canvas.
5. Export **PDF reports** or **SVG skeletons** using the buttons.
6. Click thumbnails to view full-size images.

---

## 📂 Folder Structure

```
fingerprint-matcher-v7/
│
├─ fingerprint_app_v7.py       # Main application
├─ README.md                   # Project overview
├─ requirements.txt            # Python dependencies
├─ sample_images/              # Optional: sample query and target fingerprints
└─ reports/                    # Generated PDF reports and SVGs
```

---

## ⚡ Match Visualization

* Matched keypoints shown as **green bubbles**
* Low matches / mismatches indicated with **red X**
* Skeleton images provide clear ridge structure analysis

---

## 🎯 Use Cases

* Biometrics verification
* Forensics & security
* Academic research in fingerprint analysis
* Identity verification systems

---



## 📬 Contact

Created by **\[Mubashir abbas ]** – [LinkedIn](https://www.linkedin.com/in/mubashirabbas/) | [GitHub](https://github.com/mubashirabbass)
mubashirabbasedu12@gmail.com

Feel free to open issues, suggest features, or contribute!

```

 
