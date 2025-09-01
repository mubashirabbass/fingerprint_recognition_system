# fingerprint_app_v7.py
import os
import threading
import time
import math
import io
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import svgwrite
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# -------------------------
# Utilities / Preprocessing
# -------------------------
def load_image_any(path):
    """Load image robustly (color or grayscale)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    # If image has alpha channel, drop it
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def auto_crop(gray):
    """Crop to largest contour (fingerprint region)."""
    g = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = int(max(5, min(w,h)*0.05))
    x = max(0, x-pad); y = max(0, y-pad)
    w = min(gray.shape[1]-x, w+2*pad); h = min(gray.shape[0]-y, h+2*pad)
    return gray[y:y+h, x:x+w]

def enhance(gray):
    """Denoise + CLAHE"""
    den = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(den)

def get_skeleton(gray):
    """Binary + skeletonize using skimage (robust). Returns uint8 0/255 image."""
    thresh = threshold_otsu(gray)
    binary = (gray < thresh).astype(np.uint8)  # ridges likely darker
    sk = skeletonize(binary).astype(np.uint8) * 255
    return sk

# Zhang-Suen thinning fallback (if needed) - works on binary 0/255 images
def zhang_suen_thinning(bin_img):
    img = (bin_img > 0).astype(np.uint8)
    prev = np.zeros(img.shape, np.uint8)
    changed = True
    rows, cols = img.shape
    while True:
        to_remove = []
        # Step 1
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                P = img[i-1:i+2, j-1:j+2].flatten()
                p = P[4]
                if p != 1: continue
                neighbors = P[[1,2,5,8,7,6,3,0]]
                C = sum((neighbors[k]==0 and neighbors[(k+1)%8]==1) for k in range(8))
                N = neighbors.sum()
                if 2 <= N <= 6 and C == 1 and neighbors[0]*neighbors[2]*neighbors[4]==0 and neighbors[2]*neighbors[4]*neighbors[6]==0:
                    to_remove.append((i,j))
        for (i,j) in to_remove:
            img[i,j] = 0
        to_remove = []
        # Step 2
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                P = img[i-1:i+2, j-1:j+2].flatten()
                p = P[4]
                if p != 1: continue
                neighbors = P[[1,2,5,8,7,6,3,0]]
                C = sum((neighbors[k]==0 and neighbors[(k+1)%8]==1) for k in range(8))
                N = neighbors.sum()
                if 2 <= N <= 6 and C == 1 and neighbors[0]*neighbors[2]*neighbors[6]==0 and neighbors[0]*neighbors[4]*neighbors[6]==0:
                    to_remove.append((i,j))
        for (i,j) in to_remove:
            img[i,j] = 0
        if len(to_remove) == 0:
            break
    return (img * 255).astype(np.uint8)

# -------------------------
# Feature Extraction & Matching
# -------------------------
def extract_orb_features(img_gray, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(img_gray, None)
    return kp, des

def match_and_ransac(kp1, des1, kp2, des2):
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return [], None, None, None  # no matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    # ratio test
    good = []
    for m,n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 4:
        return good, None, None, None
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=2000, confidence=0.99)
    if mask is None:
        mask = np.zeros((len(good),1), dtype=np.uint8)
    inliers = [g for g,m in zip(good, mask.ravel()) if m==1]
    return good, inliers, M, mask

# -------------------------
# Visualization creation
# -------------------------
def make_match_visualization(img1_gray, img2_gray, kp1, kp2, good_matches, inlier_matches, save_path=None, miss_cross=False):
    """
    Create an image like your example:
    - side-by-side original grayscale images
    - draw keypoints as colored circles
    - draw one or more lines connecting inlier matches (blue)
    - if miss_cross True -> draw big red X on parts or center to indicate mismatch
    """
    # create RGB copies
    a = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2RGB)
    b = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2RGB)
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1]] = b

    # choose many colors for keypoints
    rng = np.random.default_rng(12345)
    colors = [tuple(int(c) for c in rng.integers(0,256,3)) for _ in range(len(kp1)+len(kp2)+50)]

    # Draw all keypoints (small circles) - different colors for each match pair when possible
    for i,k in enumerate(kp1):
        x,y = int(k.pt[0]), int(k.pt[1])
        cv2.circle(canvas, (x, y), 4, colors[i%len(colors)], 2)
    for j,k in enumerate(kp2):
        x,y = int(k.pt[0]) + a.shape[1], int(k.pt[1])
        cv2.circle(canvas, (x, y), 4, colors[(j+len(kp1))%len(colors)], 2)

    # Draw inlier lines (use distinct color)
    for m in inlier_matches:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        # shift pt2 x by width of left image
        pt2_shift = (int(pt2[0] + a.shape[1]), int(pt2[1]))
        cv2.line(canvas, pt1, pt2_shift, (10, 10, 200), 2)

    # If mismatch (no inliers), draw large red X over each thumbnail
    if miss_cross:
        # left thumb bbox
        lw, lh = a.shape[1], a.shape[0]
        # draw X on left area
        cv2.line(canvas, (10,10), (lw-10, lh-10), (0,0,255), 6)
        cv2.line(canvas, (lw-10,10), (10, lh-10), (0,0,255), 6)
        # right area offset
        ox = a.shape[1]
        cv2.line(canvas, (ox+10,10), (ox+b.shape[1]-10, b.shape[0]-10), (0,0,255), 6)
        cv2.line(canvas, (ox+b.shape[1]-10,10), (ox+10, b.shape[0]-10), (0,0,255), 6)

    if save_path:
        cv2.imencode('.png', canvas)[1].tofile(save_path)
    return canvas

# -------------------------
# SVG export from skeleton
# -------------------------
def save_skeleton_svg(skel_img, out_svg_path):
    # Find contours from skeleton (thin lines)
    # We will draw polylines for each contour
    # First get binary
    bin_img = (skel_img > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = bin_img.shape
    dwg = svgwrite.Drawing(out_svg_path, size=(w, h))
    for c in cnts:
        pts = [(int(pt[0][0]), int(pt[0][1])) for pt in c]
        if len(pts) > 2:
            dwg.add(dwg.polyline(pts, stroke='black', fill='none', stroke_width=1))
    dwg.save()

# -------------------------
# PDF report generation
# -------------------------
def save_pdf_report(out_pdf_path, img_query_gray, img_target_gray, skeleton_q, skeleton_t, match_vis_path, match_percent, decision_text, stats:dict):
    c = canvas.Canvas(out_pdf_path, pagesize=A4)
    w, h = A4
    margin = 40
    y = h - margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "Fingerprint Matching Report")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    c.drawString(margin, y, f"Decision: {decision_text}   |   Match %: {match_percent:.2f}")
    if stats:
        y -= 18
        c.drawString(margin, y, f"Stats: {stats}")
    y -= 30

    # Insert match visualization (full width)
    if match_vis_path and os.path.exists(match_vis_path):
        img = ImageReader(match_vis_path)
        # scale to page width minus margins
        max_w = w - 2*margin
        max_h = 320
        c.drawImage(img, margin, y - max_h, width=max_w, height=max_h)
        y -= (max_h + 10)

    # Then small thumbnails and skeletons
    # Query original
    qtmp = "tmp_query.png"
    ttmp = "tmp_target.png"
    sq = "tmp_skel_q.png"
    st = "tmp_skel_t.png"
    cv2.imencode('.png', img_query_gray)[1].tofile(qtmp)
    cv2.imencode('.png', img_target_gray)[1].tofile(ttmp)
    cv2.imencode('.png', skeleton_q)[1].tofile(sq)
    cv2.imencode('.png', skeleton_t)[1].tofile(st)

    # place them side-by-side
    thumb_w = (w - 3*margin)/2
    thumb_h = thumb_w
    c.drawImage(ImageReader(qtmp), margin, y - thumb_h, width=thumb_w, height=thumb_h)
    c.drawImage(ImageReader(ttmp), margin + thumb_w + margin, y - thumb_h, width=thumb_w, height=thumb_h)
    y -= (thumb_h + 10)
    c.drawString(margin, y, "Originals (left=query, right=target)")
    y -= 20

    c.drawImage(ImageReader(sq), margin, y - thumb_h, width=thumb_w, height=thumb_h)
    c.drawImage(ImageReader(st), margin + thumb_w + margin, y - thumb_h, width=thumb_w, height=thumb_h)
    y -= (thumb_h + 10)
    c.drawString(margin, y, "Skeletons (left=query, right=target)")
    y -= 30

    c.showPage()
    c.save()
    # cleanup tmp files
    for f in [qtmp, ttmp, sq, st]:
        try:
            os.remove(f)
        except Exception:
            pass

# -------------------------
# GUI Application
# -------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Fingerprint Matcher — Visual Report & SVG export")
        root.geometry("1150x800")

        self.query_path = None
        self.target_path = None

        self.gray_q = None
        self.gray_t = None
        self.skel_q = None
        self.skel_t = None
        self.kp_q = None
        self.kp_t = None
        self.des_q = None
        self.des_t = None
        self.good = None
        self.inliers = None
        self.M = None
        self.inlier_mask = None
        self.match_vis = None
        self.match_percent = 0.0
        self.decision = "N/A"
        self.stats = {}

        # Top controls
        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(fill='x')

        ttk.Button(ctrl, text="Load Query", command=self.load_query).grid(row=0, column=0, padx=6)
        ttk.Button(ctrl, text="Load Target", command=self.load_target).grid(row=0, column=1, padx=6)
        self.btn_match = ttk.Button(ctrl, text="Match", command=self.match_thread_start)
        self.btn_match.grid(row=0, column=2, padx=6)
        self.progress = ttk.Progressbar(ctrl, mode='indeterminate', length=200)
        self.progress.grid(row=0, column=3, padx=10)
        self.status_label = ttk.Label(ctrl, text="Ready")
        self.status_label.grid(row=0, column=4, padx=6)

        ttk.Button(ctrl, text="Save Report (PDF)", command=self.save_report_dialog).grid(row=0, column=5, padx=8)
        ttk.Button(ctrl, text="Export SVGs", command=self.export_svgs_dialog).grid(row=0, column=6, padx=8)

        # Middle frames: thumbnails + match preview
        mid = ttk.Frame(root, padding=8)
        mid.pack(fill='both', expand=True)

        left = ttk.Frame(mid)
        left.pack(side='left', fill='y', padx=10)

        lab_q = ttk.Label(left, text="Query (click to open full)")
        lab_q.pack()
        self.thumb_q = tk.Label(left, background="#222", width=200, height=200)
        self.thumb_q.pack(pady=6)
        self.thumb_q.bind("<Button-1>", lambda e: self.open_full(self.query_path))

        lab_t = ttk.Label(left, text="Target (click to open full)")
        lab_t.pack()
        self.thumb_t = tk.Label(left, background="#222", width=200, height=200)
        self.thumb_t.pack(pady=6)
        self.thumb_t.bind("<Button-1>", lambda e: self.open_full(self.target_path))

        # center: match visualization
        center = ttk.Frame(mid)
        center.pack(side='left', expand=True, fill='both', padx=6)
        ttk.Label(center, text="Match Preview").pack()
        self.preview_canvas = tk.Canvas(center, width=760, height=520, bg="#111")
        self.preview_canvas.pack(pady=6)
        # status / stats
        self.result_var = tk.StringVar(value="Result: N/A")
        ttk.Label(center, textvariable=self.result_var, font=('Segoe UI', 12)).pack(pady=4)

    # ---------- UI helpers ----------
    def show_thumbnail(self, pil_img, widget, size=(200,200)):
        # pil_img is PIL.Image
        img = pil_img.copy()
        img.thumbnail(size)
        tkimg = ImageTk.PhotoImage(img)
        widget.image = tkimg
        widget.config(image=tkimg)

    def open_full(self, path):
        if not path or not os.path.exists(path):
            messagebox.showinfo("Full image", "No image available")
            return
        top = tk.Toplevel(self.root)
        top.title(os.path.basename(path))
        img_cv = load_image_any(path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) if img_cv.ndim==3 else cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(img_rgb)
        tkimg = ImageTk.PhotoImage(pil)
        lbl = tk.Label(top, image=tkimg)
        lbl.image = tkimg
        lbl.pack()

    # ---------- load images ----------
    def load_query(self):
        p = filedialog.askopenfilename(title="Select Query fingerprint",
                                       filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.*")])
        if not p:
            return
        try:
            img = load_image_any(p)
        except Exception as e:
            messagebox.showerror("Load error", str(e)); return
        gray = to_gray(img)
        gray = auto_crop(gray)
        gray = enhance(gray)
        self.query_path = p
        self.gray_q = cv2.resize(gray, (400,400), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(self.gray_q)
        self.show_thumbnail(pil, self.thumb_q)
        self.skel_q = get_skeleton(self.gray_q)

    def load_target(self):
        p = filedialog.askopenfilename(title="Select Target fingerprint",
                                       filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.*")])
        if not p:
            return
        try:
            img = load_image_any(p)
        except Exception as e:
            messagebox.showerror("Load error", str(e)); return
        gray = to_gray(img)
        gray = auto_crop(gray)
        gray = enhance(gray)
        self.target_path = p
        self.gray_t = cv2.resize(gray, (400,400), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(self.gray_t)
        self.show_thumbnail(pil, self.thumb_t)
        self.skel_t = get_skeleton(self.gray_t)

    # ---------- matching (threaded) ----------
    def match_thread_start(self):
        if not self.query_path or not self.target_path:
            messagebox.showerror("Error", "Load both fingerprints first")
            return
        # start progress
        self.progress_start()
        th = threading.Thread(target=self._match_worker, daemon=True)
        th.start()

    def progress_start(self):
        self.progress.start(10)
        self.status_label.config(text="Matching...")
        self.btn_match.state(['disabled'])

    def progress_stop(self):
        self.progress.stop()
        self.status_label.config(text="Ready")
        self.btn_match.state(['!disabled'])

    def _match_worker(self):
        try:
            # extract features on skeleton images (better)
            sk1 = self.skel_q if self.skel_q is not None else get_skeleton(self.gray_q)
            sk2 = self.skel_t if self.skel_t is not None else get_skeleton(self.gray_t)

            kp1, des1 = extract_orb_features(sk1, nfeatures=3000)
            kp2, des2 = extract_orb_features(sk2, nfeatures=3000)
            good, inliers, M, mask = match_and_ransac(kp1, des1, kp2, des2)
            self.kp_q, self.kp_t, self.des_q, self.des_t = kp1, kp2, des1, des2
            self.good = good
            self.inliers = inliers
            self.M = M
            self.inlier_mask = mask

            # compute match percent: inliers relative to avg keypoints (robust)
            num_inliers = len(inliers) if inliers is not None else 0
            denom = max(1.0, (len(kp1) + len(kp2)) / 2.0)
            percent = (num_inliers / denom) * 100.0
            self.match_percent = percent
            self.stats = {"kp1": len(kp1), "kp2": len(kp2), "good": len(good), "inliers": num_inliers}

            # create visualization image (save to temp)
            miss = (num_inliers < 4)  # treat as mismatch when too few inliers
            vis_path = os.path.join(os.getcwd(), "match_visual_temp.png")
            vis = make_match_visualization(self.gray_q, self.gray_t, kp1, kp2, good, inliers if inliers else [], save_path=vis_path, miss_cross=miss)

            self.match_vis = vis
            self.decision = "MATCH" if (percent >= 40.0 and not miss) else "NOT MATCH"

            # update UI in main thread
            self.root.after(0, self._match_done_ui, vis_path)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error during matching", str(e)))
            self.root.after(0, self.progress_stop)

    def _match_done_ui(self, vis_path):
        # show preview image in center canvas
        try:
            vis_cv = cv2.imdecode(np.fromfile(vis_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if vis_cv is None:
                raise RuntimeError("Could not load visualization")
            vis_rgb = cv2.cvtColor(vis_cv, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(vis_rgb)
            pil.thumbnail((760, 520))
            tkimg = ImageTk.PhotoImage(pil)
            self.preview_canvas.delete('all')
            self.preview_canvas.create_image(380, 260, image=tkimg)
            self.preview_canvas.image = tkimg
            # update result text
            self.result_var.set(f"{self.decision} — {self.match_percent:.1f}%  (kp1:{self.stats['kp1']} kp2:{self.stats['kp2']} good:{self.stats['good']} inliers:{self.stats['inliers']})")
        except Exception as e:
            messagebox.showwarning("Preview error", str(e))
        finally:
            self.progress_stop()

    # ---------- save report / svgs ----------
    def save_report_dialog(self):
        if self.match_vis is None:
            messagebox.showerror("Error", "No match result to save. Run matching first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF","*.pdf")], title="Save report PDF")
        if not out:
            return
        # save temp match visualization as PNG first
        vis_tmp = os.path.splitext(out)[0] + "_vis.png"
        cv2.imencode('.png', self.match_vis)[1].tofile(vis_tmp)
        # write pdf
        save_pdf_report(out, self.gray_q, self.gray_t, self.skel_q, self.skel_t, vis_tmp, self.match_percent, self.decision, self.stats)
        messagebox.showinfo("Saved", f"Report saved: {out}")
        try:
            os.remove(vis_tmp)
        except Exception:
            pass

    def export_svgs_dialog(self):
        if self.skel_q is None or self.skel_t is None:
            messagebox.showerror("Error", "No skeletons available. Load images and match first.")
            return
        folder = filedialog.askdirectory(title="Choose folder to save SVGs")
        if not folder:
            return
        qsvg = os.path.join(folder, "query_skeleton.svg")
        tsvg = os.path.join(folder, "target_skeleton.svg")
        save_skeleton_svg(self.skel_q, qsvg)
        save_skeleton_svg(self.skel_t, tsvg)
        messagebox.showinfo("Saved", f"SVGs saved:\n{qsvg}\n{tsvg}")

# -------------------------
# Entry point
# -------------------------
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
