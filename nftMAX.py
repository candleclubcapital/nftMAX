# art_pipeline_gui.py
import os
import re
import sys
import math
import json
import time
import random
import threading
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Set

# ---------- Third-party ----------
# GUI
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QGridLayout, QTabWidget, QMessageBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import QThread, Signal, Qt

# Imaging
from PIL import Image, ImageFilter

# Background removal (optional)
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

# Selenium (scraper)
import requests
from urllib.parse import urlparse, parse_qs, unquote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
try:
    import undetected_chromedriver as uc
    HAS_UC = True
except ImportError:
    HAS_UC = False


# ===========================
# Shared Utils
# ===========================

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name).strip("_ ").replace(" ", "_")

def list_images_recursive(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]

def has_meaningful_alpha(img: Image.Image, threshold: int = 5) -> bool:
    if img.mode != "RGBA":
        return False
    alpha = img.split()[-1]
    if alpha.getbbox() is None:
        return False
    w, h = alpha.size
    samples = 0
    nonfull = 0
    step_x = max(1, w // 64)
    step_y = max(1, h // 64)
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            a = alpha.getpixel((x, y))
            samples += 1
            if a < 250:
                nonfull += 1
                if nonfull >= threshold:
                    return True
    return False

def auto_remove_bg(img: Image.Image) -> Image.Image:
    if not HAS_REMBG:
        return img.convert("RGBA")
    try:
        with BytesIO() as buf_in:
            img.convert("RGBA").save(buf_in, format="PNG")
            data = buf_in.getvalue()
        out = rembg_remove(data)
        return Image.open(BytesIO(out)).convert("RGBA")
    except Exception:
        return img.convert("RGBA")

def random_scale_to_canvas(img: Image.Image, canvas_size: Tuple[int, int],
                           min_scale: float, max_scale: float) -> Image.Image:
    W, H = canvas_size
    iw, ih = img.size
    base = min(W, H)
    s = random.uniform(min_scale, max_scale)
    target_long = max(64, int(base * s))
    if iw >= ih:
        scale = target_long / float(iw)
    else:
        scale = target_long / float(ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    return img.resize((nw, nh), Image.LANCZOS)

def place_random_position(canvas_size: Tuple[int, int], img_size: Tuple[int, int], margin: int) -> Tuple[int, int]:
    W, H = canvas_size
    w, h = img_size
    x = random.randint(-margin, max(-margin, W - w + margin))
    y = random.randint(-margin, max(-margin, H - h + margin))
    return x, y

def paste_with_alpha(base: Image.Image, overlay: Image.Image, xy: Tuple[int, int]):
    base.alpha_composite(overlay, dest=xy)

def add_shadow(overlay: Image.Image, blur_radius: int = 10, opacity: int = 140, offset: Tuple[int, int] = (8, 8)) -> Image.Image:
    ox, oy = offset
    w, h = overlay.size
    shadow = Image.new("RGBA", (w + abs(ox), h + abs(oy)), (0, 0, 0, 0))
    alpha = overlay.split()[-1]
    shadow_alpha = Image.new("L", shadow.size, 0)
    shadow_alpha.paste(alpha, (max(0, ox), max(0, oy)))
    shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(blur_radius))
    shadow_rgba = Image.merge("RGBA", (
        Image.new("L", shadow.size, 0),
        Image.new("L", shadow.size, 0),
        Image.new("L", shadow.size, 0),
        shadow_alpha.point(lambda a: min(opacity, a))
    ))
    combined = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
    combined.alpha_composite(shadow_rgba, (0, 0))
    combined.alpha_composite(overlay, (max(0, -ox), max(0, -oy)))
    return combined

def guess_extension(url: str, content_type: str) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "gif" in ct: return ".gif"
    if "webp" in ct: return ".webp"
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


# ===========================
# SCRAPER (Bing + DDG)
# ===========================

class ImageScraperWorker(QThread):
    log = Signal(str)
    finished = Signal()
    progress = Signal(str, int, int)

    def __init__(self, keywords: List[str], target: int, out_dir: str,
                 provider: str = "Auto", headless: bool = True):
        super().__init__()
        self.keywords = [k.strip() for k in keywords if k.strip()]
        self.target = target
        self.out_dir = out_dir
        self.provider = provider
        self.headless = headless
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _make_driver(self):
        if HAS_UC:
            opts = uc.ChromeOptions()
            if self.headless:
                opts.add_argument("--headless=new")
            opts.add_argument(f"--user-agent={USER_AGENT}")
            opts.add_argument("--disable-gpu")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--window-size=1280,1600")
            driver = uc.Chrome(options=opts)
            driver.set_page_load_timeout(60)
            return driver
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument(f"--user-agent={USER_AGENT}")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1280,1600")
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(60)
        return driver

    def _bing_urls(self, driver, keyword: str, target: int) -> List[str]:
        search_url = f"https://www.bing.com/images/search?q={requests.utils.quote(keyword)}&form=HDRSC2&first=1"
        self.log.emit(f"→ Bing: {search_url}")
        driver.get(search_url)
        time.sleep(1.5)
        urls, seen = [], set()

        def harvest():
            nonlocal urls
            tiles = driver.find_elements(By.CSS_SELECTOR, "a.iusc, div.iusc")
            for t in tiles:
                try:
                    m = t.get_attribute("m")
                    if not m: continue
                    data = json.loads(m)
                    u = data.get("murl") or data.get("purl")
                    if u and u.startswith("http") and u not in seen:
                        seen.add(u)
                        urls.append(u)
                        if len(urls) >= target:
                            break
                except Exception:
                    continue

        harvest()
        stagnant = 0
        while len(urls) < target and not self._stop.is_set():
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(1.0)
            before = len(urls)
            harvest()
            self.progress.emit(keyword, len(urls), target)
            stagnant = stagnant + 1 if len(urls) == before else 0
            if stagnant > 5:
                break
        return urls[:target]

    def _ddg_urls(self, driver, keyword: str, target: int) -> List[str]:
        url = f"https://duckduckgo.com/?q={requests.utils.quote(keyword)}&ia=images&iar=images"
        self.log.emit(f"→ DuckDuckGo: {url}")
        driver.get(url)
        time.sleep(1.5)
        urls, seen = [], set()

        def harvest():
            nonlocal urls
            imgs = driver.find_elements(By.CSS_SELECTOR, "img.tile--img__img, img[loading='lazy']")
            for im in imgs:
                try:
                    src = im.get_attribute("src") or im.get_attribute("data-src")
                    if not src: continue
                    if "duckduckgo.com/iu/?" in src:
                        q = parse_qs(urlparse(src).query)
                        orig = unquote(q.get("u", [""])[0])
                        if orig and orig.startswith("http") and orig not in seen:
                            seen.add(orig)
                            urls.append(orig)
                    elif src.startswith("http") and "data:image" not in src:
                        if src not in seen:
                            seen.add(src)
                            urls.append(src)
                    if len(urls) >= target:
                        break
                except Exception:
                    continue

        harvest()
        stagnant = 0
        while len(urls) < target and not self._stop.is_set():
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(1.0)
            before = len(urls)
            harvest()
            self.progress.emit(keyword, len(urls), target)
            stagnant = stagnant + 1 if len(urls) == before else 0
            if stagnant > 8:
                break
        return urls[:target]

    def _collect(self, driver, keyword: str, target: int) -> List[str]:
        if self.provider == "Bing":
            return self._bing_urls(driver, keyword, target)
        if self.provider == "DuckDuckGo":
            return self._ddg_urls(driver, keyword, target)
        urls = self._bing_urls(driver, keyword, target)
        if len(urls) < target // 2:
            self.log.emit("!! Bing low results, falling back to DuckDuckGo…")
            urls.extend(self._ddg_urls(driver, keyword, target - len(urls)))
        return urls[:target]

    def run(self):
        ensure_dir(self.out_dir)
        session = requests.Session()
        try:
            driver = self._make_driver()
        except Exception as e:
            self.log.emit(f"!! WebDriver error: {e}")
            self.finished.emit()
            return
        try:
            for kw in self.keywords:
                if self._stop.is_set(): break
                out_folder = os.path.join(self.out_dir, sanitize_filename(kw))
                ensure_dir(out_folder)
                self.log.emit(f"\n=== {kw} ===")
                urls = self._collect(driver, kw, self.target)
                self.log.emit(f"Found {len(urls)} URLs for {kw}")
                count = 0
                for idx, u in enumerate(urls, 1):
                    if self._stop.is_set(): break
                    base = os.path.join(out_folder, f"{sanitize_filename(kw)}_{idx:04d}")
                    try:
                        r = session.get(u, headers={"User-Agent": USER_AGENT}, timeout=20, stream=True)
                        r.raise_for_status()
                        ext = guess_extension(u, r.headers.get("Content-Type", ""))
                        dest = base + ext
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                        count += 1
                        self.log.emit(f"✔ {dest}")
                    except Exception as e:
                        self.log.emit(f"-- skip {u} ({e})")
                    time.sleep(0.15)
                self.log.emit(f"Done {kw}: {count} saved.")
        finally:
            try: driver.quit()
            except Exception: pass
            self.finished.emit()


# ===========================
# EXTRACTOR (rembg-based)
# ===========================

class ExtractWorker(QThread):
    log = Signal(str)
    finished = Signal()

    def __init__(self, input_dir, output_dir, out_w, out_h):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.out_w = out_w
        self.out_h = out_h
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        if not self.input_dir.exists():
            self.log.emit("!! Input folder not found.")
            self.finished.emit()
            return

        images = list_images_recursive(self.input_dir)
        self.log.emit(f"Found {len(images)} images.")

        for idx, img_path in enumerate(images, 1):
            if self._stop: break
            try:
                self.log.emit(f"[{idx}/{len(images)}] {img_path}")
                with open(img_path, "rb") as f:
                    input_bytes = f.read()

                # Background removal to alpha PNG
                if not HAS_REMBG:
                    self.log.emit("   (rembg not installed — copying with alpha if any)")
                    img = Image.open(BytesIO(input_bytes)).convert("RGBA")
                    out_img = img
                else:
                    out_bytes = rembg_remove(input_bytes)
                    out_img = Image.open(BytesIO(out_bytes)).convert("RGBA")

                # Resize to requested resolution (ignore aspect as requested)
                out_img = out_img.resize((self.out_w, self.out_h), Image.LANCZOS)

                # Mirror input structure
                rel = img_path.relative_to(self.input_dir)
                out_file = self.output_dir / rel.with_suffix(".png")
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_img.save(out_file, "PNG")
                self.log.emit(f"✔ {out_file}")
            except Exception as e:
                self.log.emit(f"!! Failed {img_path}: {e}")

        self.finished.emit()


# ===========================
# COLLAGE MAKER
# ===========================

class CollageWorker(QThread):
    log = Signal(str)
    finished = Signal()

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 out_w: int,
                 out_h: int,
                 images_per_collage: int,
                 collage_count: int,
                 min_scale: float,
                 max_scale: float,
                 max_rotation_deg: float,
                 allow_reuse_if_insufficient: bool,
                 margin: int,
                 background_mode: str,
                 bg_color_hex: str,
                 add_drop_shadow: bool,
                 use_bg_removal_if_no_alpha: bool,
                 placement: str,
                 seed: Optional[int]):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.out_w = out_w
        self.out_h = out_h
        self.images_per_collage = images_per_collage
        self.collage_count = collage_count
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_rotation_deg = max_rotation_deg
        self.allow_reuse_if_insufficient = allow_reuse_if_insufficient
        self.margin = margin
        self.background_mode = background_mode
        self.bg_color_hex = bg_color_hex
        self.add_drop_shadow = add_drop_shadow
        self.use_bg_removal_if_no_alpha = use_bg_removal_if_no_alpha
        self.placement = placement
        self.seed = seed
        self._stop = False

    def stop(self):
        self._stop = True

    def _make_canvas(self) -> Image.Image:
        if self.background_mode == "Transparent":
            return Image.new("RGBA", (self.out_w, self.out_h), (0, 0, 0, 0))
        hexv = self.bg_color_hex.strip().lstrip("#")
        if len(hexv) == 3: hexv = "".join([c*2 for c in hexv])
        try:
            r = int(hexv[0:2], 16); g = int(hexv[2:4], 16); b = int(hexv[4:6], 16)
        except Exception:
            r, g, b = 10, 10, 10
        return Image.new("RGBA", (self.out_w, self.out_h), (r, g, b, 255))

    def _choose_positions(self, count: int) -> List[Tuple[int, int]]:
        positions: List[Tuple[int,int]] = []
        if self.placement == "Grid-ish":
            cols = max(1, int(math.sqrt(count)))
            rows = max(1, math.ceil(count / cols))
            cell_w = max(1, self.out_w // cols)
            cell_h = max(1, self.out_h // rows)
            for r in range(rows):
                for c in range(cols):
                    if len(positions) >= count: break
                    cx = c * cell_w + cell_w // 2
                    cy = r * cell_h + cell_h // 2
                    jitter_x = random.randint(-cell_w // 4, cell_w // 4)
                    jitter_y = random.randint(-cell_h // 4, cell_h // 4)
                    x = max(-self.margin, min(self.out_w - 1, cx + jitter_x))
                    y = max(-self.margin, min(self.out_h - 1, cy + jitter_y))
                    positions.append((x, y))
            random.shuffle(positions)
            return positions
        if self.placement == "Center Bias":
            for _ in range(count):
                mu_x, mu_y = self.out_w/2, self.out_h/2
                sigma_x, sigma_y = self.out_w/6, self.out_h/6
                x = int(random.gauss(mu_x, sigma_x))
                y = int(random.gauss(mu_y, sigma_y))
                positions.append((x, y))
            random.shuffle(positions)
            return positions
        for _ in range(count):
            positions.append(place_random_position((self.out_w, self.out_h), (0, 0), self.margin))
        return positions

    def run(self):
        if self.seed is not None:
            random.seed(self.seed)

        if not self.input_dir.exists():
            self.log.emit("!! Collage input dir does not exist.")
            self.finished.emit()
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_imgs = list_images_recursive(self.input_dir)
        if not all_imgs:
            self.log.emit("!! No images found for collage.")
            self.finished.emit()
            return

        required_unique = self.images_per_collage * self.collage_count
        if len(all_imgs) < required_unique and not self.allow_reuse_if_insufficient:
            self.log.emit(f"!! Not enough unique images ({len(all_imgs)}) for {self.collage_count}×{self.images_per_collage}.")
            max_collages = len(all_imgs) // max(1, self.images_per_collage)
            self.collage_count = max_collages
            self.log.emit(f"→ Adjusted collage_count to {self.collage_count}")

        random.shuffle(all_imgs)
        used: Set[Path] = set()
        produced = 0
        idx_global = 0

        for ci in range(self.collage_count):
            if self._stop: break
            canvas = self._make_canvas()

            chosen: List[Path] = []
            attempts = 0
            while len(chosen) < self.images_per_collage and attempts < 10_000:
                attempts += 1
                remaining = [p for p in all_imgs if (self.allow_reuse_if_insufficient or p not in used)]
                if not remaining: break
                p = random.choice(remaining)
                if not self.allow_reuse_if_insufficient and p in used:
                    continue
                chosen.append(p)
                if not self.allow_reuse_if_insufficient:
                    used.add(p)

            if not chosen:
                self.log.emit(f"-- Skip collage {ci+1}: no images left.")
                continue

            positions = self._choose_positions(len(chosen))

            for i, img_path in enumerate(chosen):
                if self._stop: break
                self.log.emit(f"[{ci+1}/{self.collage_count}] {i+1}/{len(chosen)}: {img_path.name}")
                try:
                    img = Image.open(img_path).convert("RGBA")
                except Exception:
                    self.log.emit(f"   !! open failed: {img_path}")
                    continue

                if self.use_bg_removal_if_no_alpha and not has_meaningful_alpha(img):
                    img = auto_remove_bg(img)

                img = random_scale_to_canvas(img, (self.out_w, self.out_h), self.min_scale, self.max_scale)
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

                if self.add_drop_shadow:
                    img = add_shadow(img, blur_radius=10, opacity=140, offset=(8, 8))

                if self.placement == "Random":
                    xy = place_random_position((self.out_w, self.out_h), img.size, self.margin)
                else:
                    if positions:
                        cx, cy = positions.pop(0)
                    else:
                        cx, cy = place_random_position((self.out_w, self.out_h), img.size, self.margin)
                    xy = (int(cx - img.width/2), int(cy - img.height/2))

                paste_with_alpha(canvas, img, xy)

            idx_global += 1
            out_path = self.output_dir / f"collage_{idx_global:04d}.png"
            try:
                canvas.save(out_path, "PNG")
                produced += 1
                self.log.emit(f"✔ Saved: {out_path}")
            except Exception as e:
                self.log.emit(f"!! Save failed: {e}")

        self.log.emit(f"=== Collages done: {produced} ===")
        self.finished.emit()


# ===========================
# MAIN GUI (Tabs + Pipeline)
# ===========================

class PipelineGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Art Pipeline — Scrape → Extract → Collage")
        self.setGeometry(80, 80, 1060, 760)

        self.scrape_worker: Optional[ImageScraperWorker] = None
        self.extract_worker: Optional[ExtractWorker] = None
        self.collage_worker: Optional[CollageWorker] = None

        root = QVBoxLayout(self)

        # Tabs
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        # ----- SCRAPER TAB -----
        self.scrape_tab = QWidget()
        self.tabs.addTab(self.scrape_tab, "1) Scrape")
        self._build_scrape_tab()

        # ----- EXTRACT TAB -----
        self.extract_tab = QWidget()
        self.tabs.addTab(self.extract_tab, "2) Extract")
        self._build_extract_tab()

        # ----- COLLAGE TAB -----
        self.collage_tab = QWidget()
        self.tabs.addTab(self.collage_tab, "3) Collage")
        self._build_collage_tab()

        # ----- PIPELINE CONTROLS -----
        pipe_box = QGroupBox("Pipeline (Run All)")
        pipe_layout = QGridLayout(pipe_box)

        self.pipe_keywords = QLineEdit("cat, rabbit")
        self.pipe_target = QSpinBox(); self.pipe_target.setRange(1, 1000); self.pipe_target.setValue(25)
        self.pipe_provider = QComboBox(); self.pipe_provider.addItems(["Auto", "Bing", "DuckDuckGo"])
        self.pipe_headless = QCheckBox("Headless"); self.pipe_headless.setChecked(True)
        self.pipe_scrape_out = QLineEdit(str(Path.cwd() / "scraped_images"))
        self.btn_pipe_scrape_out = QPushButton("Browse…")
        self.btn_pipe_scrape_out.clicked.connect(lambda: self._choose_dir_into(self.pipe_scrape_out))

        self.pipe_extract_in = QLineEdit(self.pipe_scrape_out.text())
        self.pipe_extract_out = QLineEdit(str(Path.cwd() / "extracted"))
        self.btn_pipe_extract_in = QPushButton("Browse…")
        self.btn_pipe_extract_out = QPushButton("Browse…")
        self.btn_pipe_extract_in.clicked.connect(lambda: self._choose_dir_into(self.pipe_extract_in))
        self.btn_pipe_extract_out.clicked.connect(lambda: self._choose_dir_into(self.pipe_extract_out))
        self.pipe_ex_w = QSpinBox(); self.pipe_ex_w.setRange(16, 4096); self.pipe_ex_w.setValue(512)
        self.pipe_ex_h = QSpinBox(); self.pipe_ex_h.setRange(16, 4096); self.pipe_ex_h.setValue(512)

        self.pipe_col_in = QLineEdit(self.pipe_extract_out.text())
        self.btn_pipe_col_in = QPushButton("Browse…")
        self.btn_pipe_col_in.clicked.connect(lambda: self._choose_dir_into(self.pipe_col_in))
        self.pipe_col_out = QLineEdit(str(Path.cwd() / "collages_out"))
        self.btn_pipe_col_out = QPushButton("Browse…")
        self.btn_pipe_col_out.clicked.connect(lambda: self._choose_dir_into(self.pipe_col_out))
        self.pipe_W = QSpinBox(); self.pipe_W.setRange(128, 8192); self.pipe_W.setValue(1920)
        self.pipe_H = QSpinBox(); self.pipe_H.setRange(128, 8192); self.pipe_H.setValue(1080)
        self.pipe_K = QSpinBox(); self.pipe_K.setRange(1, 200); self.pipe_K.setValue(18)
        self.pipe_N = QSpinBox(); self.pipe_N.setRange(1, 10000); self.pipe_N.setValue(10)
        self.pipe_minS = QDoubleSpinBox(); self.pipe_minS.setRange(0.02, 2.0); self.pipe_minS.setSingleStep(0.01); self.pipe_minS.setValue(0.18)
        self.pipe_maxS = QDoubleSpinBox(); self.pipe_maxS.setRange(0.02, 3.0); self.pipe_maxS.setSingleStep(0.01); self.pipe_maxS.setValue(0.55)
        self.pipe_rot = QDoubleSpinBox(); self.pipe_rot.setRange(0.0, 180.0); self.pipe_rot.setSingleStep(1.0); self.pipe_rot.setValue(28.0)
        self.pipe_margin = QSpinBox(); self.pipe_margin.setRange(0, 2000); self.pipe_margin.setValue(60)
        self.pipe_place = QComboBox(); self.pipe_place.addItems(["Random", "Center Bias", "Grid-ish"])
        self.pipe_bg_mode = QComboBox(); self.pipe_bg_mode.addItems(["Transparent", "Solid"])
        self.pipe_bg_hex = QLineEdit("#0a0a0a")
        self.pipe_shadow = QCheckBox("Drop shadow"); self.pipe_shadow.setChecked(True)
        self.pipe_reuse = QCheckBox("Allow reuse if insufficient uniques"); self.pipe_reuse.setChecked(False)
        self.pipe_rembg = QCheckBox("BG removal for non-alpha"); self.pipe_rembg.setChecked(False)
        if not HAS_REMBG:
            self.pipe_rembg.setText("BG removal (install rembg)")
        self.pipe_seed = QLineEdit(""); self.pipe_seed.setPlaceholderText("optional int")

        # form rows
        r = 0
        pipe_layout.addWidget(QLabel("Keywords"), r, 0); pipe_layout.addWidget(self.pipe_keywords, r, 1, 1, 3); r += 1
        pipe_layout.addWidget(QLabel("Target / keyword"), r, 0); pipe_layout.addWidget(self.pipe_target, r, 1)
        pipe_layout.addWidget(QLabel("Provider"), r, 2); pipe_layout.addWidget(self.pipe_provider, r, 3); r += 1
        pipe_layout.addWidget(self.pipe_headless, r, 0)
        pipe_layout.addWidget(QLabel("Scrape Output"), r, 1); pipe_layout.addWidget(self.pipe_scrape_out, r, 2); pipe_layout.addWidget(self.btn_pipe_scrape_out, r, 3); r += 1

        pipe_layout.addWidget(QLabel("Extract In"), r, 0); pipe_layout.addWidget(self.pipe_extract_in, r, 1); pipe_layout.addWidget(self.btn_pipe_extract_in, r, 2)
        pipe_layout.addWidget(QLabel("Extract Out"), r, 3); r += 1
        pipe_layout.addWidget(self.pipe_extract_out, r, 1); pipe_layout.addWidget(self.btn_pipe_extract_out, r, 2)
        pipe_layout.addWidget(QLabel("Size W×H"), r, 3); r += 1
        pipe_layout.addWidget(self._hbox([self.pipe_ex_w, QLabel("×"), self.pipe_ex_h]), r, 1); r += 1

        pipe_layout.addWidget(QLabel("Collage In"), r, 0); pipe_layout.addWidget(self.pipe_col_in, r, 1); pipe_layout.addWidget(self.btn_pipe_col_in, r, 2)
        pipe_layout.addWidget(QLabel("Collage Out"), r, 3); r += 1
        pipe_layout.addWidget(self.pipe_col_out, r, 1); pipe_layout.addWidget(self.btn_pipe_col_out, r, 2)
        pipe_layout.addWidget(QLabel("Canvas W×H"), r, 3); r += 1
        pipe_layout.addWidget(self._hbox([self.pipe_W, QLabel("×"), self.pipe_H]), r, 1)
        pipe_layout.addWidget(QLabel("K / N"), r, 2); pipe_layout.addWidget(self._hbox([self.pipe_K, QLabel("/"), self.pipe_N]), r, 3); r += 1
        pipe_layout.addWidget(QLabel("Min / Max Scale"), r, 0); pipe_layout.addWidget(self._hbox([self.pipe_minS, QLabel("/"), self.pipe_maxS]), r, 1)
        pipe_layout.addWidget(QLabel("Max Rot"), r, 2); pipe_layout.addWidget(self.pipe_rot, r, 3); r += 1
        pipe_layout.addWidget(QLabel("Margin"), r, 0); pipe_layout.addWidget(self.pipe_margin, r, 1)
        pipe_layout.addWidget(QLabel("Placement"), r, 2); pipe_layout.addWidget(self.pipe_place, r, 3); r += 1
        pipe_layout.addWidget(QLabel("BG Mode"), r, 0); pipe_layout.addWidget(self.pipe_bg_mode, r, 1)
        pipe_layout.addWidget(QLabel("BG Hex"), r, 2); pipe_layout.addWidget(self.pipe_bg_hex, r, 3); r += 1
        pipe_layout.addWidget(self.pipe_shadow, r, 0)
        pipe_layout.addWidget(self.pipe_reuse, r, 1)
        pipe_layout.addWidget(self.pipe_rembg, r, 2)
        pipe_layout.addWidget(QLabel("Seed"), r, 3); r += 1
        pipe_layout.addWidget(self.pipe_seed, r, 1); r += 1

        self.btn_run_pipeline = QPushButton("Run Pipeline (Scrape → Extract → Collage)")
        pipe_layout.addWidget(self.btn_run_pipeline, r, 0, 1, 4)
        root.addWidget(pipe_box)

        # logs
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("QTextEdit { background:#0b0f14; color:#c8f6ff; font-family: Menlo, Monaco, Consolas, monospace; }")
        root.addWidget(self.log_box, 1)

        # wire
        self.btn_run_pipeline.clicked.connect(self.run_pipeline)

    # ---------- Build tabs ----------
    def _build_scrape_tab(self):
        lay = QVBoxLayout(self.scrape_tab)

        self.scr_kw = QTextEdit(); self.scr_kw.setPlaceholderText("cat, rabbit, solana logo")
        self.scr_kw.setFixedHeight(90)
        self.scr_target = QSpinBox(); self.scr_target.setRange(1, 1000); self.scr_target.setValue(30)
        self.scr_provider = QComboBox(); self.scr_provider.addItems(["Auto", "Bing", "DuckDuckGo"])
        self.scr_headless = QCheckBox("Headless"); self.scr_headless.setChecked(True)
        self.scr_out = QLineEdit(str(Path.cwd() / "scraped_images"))
        btn_out = QPushButton("Browse…"); btn_out.clicked.connect(lambda: self._choose_dir_into(self.scr_out))
        btn_start = QPushButton("Start"); btn_stop = QPushButton("Stop")

        form = QGridLayout()
        r = 0
        form.addWidget(QLabel("Keywords"), r, 0); form.addWidget(self.scr_kw, r, 1, 1, 3); r += 1
        form.addWidget(QLabel("Target/keyword"), r, 0); form.addWidget(self.scr_target, r, 1)
        form.addWidget(QLabel("Provider"), r, 2); form.addWidget(self.scr_provider, r, 3); r += 1
        form.addWidget(self.scr_headless, r, 0)
        form.addWidget(QLabel("Output"), r, 1); form.addWidget(self.scr_out, r, 2); form.addWidget(btn_out, r, 3); r += 1
        form.addWidget(btn_start, r, 2); form.addWidget(btn_stop, r, 3); r += 1
        lay.addLayout(form)

        btn_start.clicked.connect(self.start_scrape)
        btn_stop.clicked.connect(self.stop_scrape)

    def _build_extract_tab(self):
        lay = QVBoxLayout(self.extract_tab)

        self.ex_in = QLineEdit(str(Path.cwd() / "scraped_images"))
        self.ex_out = QLineEdit(str(Path.cwd() / "extracted"))
        btn_in = QPushButton("Browse…"); btn_in.clicked.connect(lambda: self._choose_dir_into(self.ex_in))
        btn_out = QPushButton("Browse…"); btn_out.clicked.connect(lambda: self._choose_dir_into(self.ex_out))
        self.ex_w = QSpinBox(); self.ex_w.setRange(16, 4096); self.ex_w.setValue(512)
        self.ex_h = QSpinBox(); self.ex_h.setRange(16, 4096); self.ex_h.setValue(512)
        btn_start = QPushButton("Start"); btn_stop = QPushButton("Stop")

        form = QGridLayout()
        r = 0
        form.addWidget(QLabel("Input"), r, 0); form.addWidget(self.ex_in, r, 1); form.addWidget(btn_in, r, 2); r += 1
        form.addWidget(QLabel("Output"), r, 0); form.addWidget(self.ex_out, r, 1); form.addWidget(btn_out, r, 2); r += 1
        form.addWidget(QLabel("Size W×H"), r, 0); form.addWidget(self._hbox([self.ex_w, QLabel("×"), self.ex_h]), r, 1); r += 1
        form.addWidget(btn_start, r, 1); form.addWidget(btn_stop, r, 2); r += 1
        lay.addLayout(form)

        btn_start.clicked.connect(self.start_extract)
        btn_stop.clicked.connect(self.stop_extract)

    def _build_collage_tab(self):
        lay = QVBoxLayout(self.collage_tab)

        self.col_in = QLineEdit(str(Path.cwd() / "extracted"))
        self.col_out = QLineEdit(str(Path.cwd() / "collages_out"))
        btn_in = QPushButton("Browse…"); btn_in.clicked.connect(lambda: self._choose_dir_into(self.col_in))
        btn_out = QPushButton("Browse…"); btn_out.clicked.connect(lambda: self._choose_dir_into(self.col_out))

        self.cW = QSpinBox(); self.cW.setRange(128, 8192); self.cW.setValue(1920)
        self.cH = QSpinBox(); self.cH.setRange(128, 8192); self.cH.setValue(1080)
        self.cK = QSpinBox(); self.cK.setRange(1, 200); self.cK.setValue(18)
        self.cN = QSpinBox(); self.cN.setRange(1, 10000); self.cN.setValue(10)
        self.minS = QDoubleSpinBox(); self.minS.setRange(0.02, 2.0); self.minS.setSingleStep(0.01); self.minS.setValue(0.18)
        self.maxS = QDoubleSpinBox(); self.maxS.setRange(0.02, 3.0); self.maxS.setSingleStep(0.01); self.maxS.setValue(0.55)
        self.rot = QDoubleSpinBox(); self.rot.setRange(0.0, 180.0); self.rot.setSingleStep(1.0); self.rot.setValue(28.0)
        self.margin = QSpinBox(); self.margin.setRange(0, 2000); self.margin.setValue(60)
        self.place = QComboBox(); self.place.addItems(["Random", "Center Bias", "Grid-ish"])
        self.bg_mode = QComboBox(); self.bg_mode.addItems(["Transparent", "Solid"])
        self.bg_hex = QLineEdit("#0a0a0a")
        self.shadow = QCheckBox("Drop shadow"); self.shadow.setChecked(True)
        self.reuse = QCheckBox("Allow reuse if insufficient uniques"); self.reuse.setChecked(False)
        self.rembg = QCheckBox("BG removal for non-alpha"); self.rembg.setChecked(False)
        if not HAS_REMBG:
            self.rembg.setText("BG removal (install rembg)")
        self.seed = QLineEdit(""); self.seed.setPlaceholderText("optional int")

        btn_start = QPushButton("Start"); btn_stop = QPushButton("Stop")

        form = QGridLayout()
        r = 0
        form.addWidget(QLabel("Input"), r, 0); form.addWidget(self.col_in, r, 1); form.addWidget(btn_in, r, 2); r += 1
        form.addWidget(QLabel("Output"), r, 0); form.addWidget(self.col_out, r, 1); form.addWidget(btn_out, r, 2); r += 1
        form.addWidget(QLabel("Canvas W×H"), r, 0); form.addWidget(self._hbox([self.cW, QLabel("×"), self.cH]), r, 1)
        form.addWidget(QLabel("K / N"), r, 2); form.addWidget(self._hbox([self.cK, QLabel("/"), self.cN]), r, 3); r += 1
        form.addWidget(QLabel("Min / Max Scale"), r, 0); form.addWidget(self._hbox([self.minS, QLabel("/"), self.maxS]), r, 1)
        form.addWidget(QLabel("Max Rot"), r, 2); form.addWidget(self.rot, r, 3); r += 1
        form.addWidget(QLabel("Margin"), r, 0); form.addWidget(self.margin, r, 1)
        form.addWidget(QLabel("Placement"), r, 2); form.addWidget(self.place, r, 3); r += 1
        form.addWidget(QLabel("BG Mode"), r, 0); form.addWidget(self.bg_mode, r, 1)
        form.addWidget(QLabel("BG Hex"), r, 2); form.addWidget(self.bg_hex, r, 3); r += 1
        form.addWidget(self.shadow, r, 0)
        form.addWidget(self.reuse, r, 1)
        form.addWidget(self.rembg, r, 2)
        form.addWidget(QLabel("Seed"), r, 3); r += 1
        form.addWidget(self.seed, r, 1)
        form.addWidget(btn_start, r, 2); form.addWidget(btn_stop, r, 3); r += 1
        lay.addLayout(form)

        btn_start.clicked.connect(self.start_collage)
        btn_stop.clicked.connect(self.stop_collage)

    # ---------- Handlers: Scrape ----------
    def start_scrape(self):
        text = self.scr_kw.toPlainText().strip()
        if not text:
            self._log("!! Enter keywords."); return
        keywords = [p.strip() for p in re.split(r"[,;\n]+", text) if p.strip()]
        self.scrape_worker = ImageScraperWorker(
            keywords=keywords,
            target=self.scr_target.value(),
            out_dir=self.scr_out.text().strip(),
            provider=self.scr_provider.currentText(),
            headless=self.scr_headless.isChecked()
        )
        self.scrape_worker.log.connect(self._log)
        self.scrape_worker.progress.connect(lambda kw,c,t: self._log(f"[{kw}] {c}/{t}"))
        self.scrape_worker.finished.connect(lambda: self._log("=== Scrape done ==="))
        self.scrape_worker.start()

    def stop_scrape(self):
        if self.scrape_worker:
            self.scrape_worker.stop()
            self.scrape_worker.wait()
            self.scrape_worker = None
            self._log("=== Scrape stopped ===")

    # ---------- Handlers: Extract ----------
    def start_extract(self):
        in_dir = self.ex_in.text().strip()
        out_dir = self.ex_out.text().strip()
        if not in_dir or not out_dir:
            self._log("!! Select input and output."); return
        self.extract_worker = ExtractWorker(in_dir, out_dir, self.ex_w.value(), self.ex_h.value())
        self.extract_worker.log.connect(self._log)
        self.extract_worker.finished.connect(lambda: self._log("=== Extract done ==="))
        self.extract_worker.start()

    def stop_extract(self):
        if self.extract_worker:
            self.extract_worker.stop()
            self.extract_worker.wait()
            self.extract_worker = None
            self._log("=== Extract stopped ===")

    # ---------- Handlers: Collage ----------
    def start_collage(self):
        seed = None
        if self.seed.text().strip():
            try: seed = int(self.seed.text().strip())
            except ValueError: self._log("!! Seed must be int (ignored).")
        self.collage_worker = CollageWorker(
            input_dir=self.col_in.text().strip(),
            output_dir=self.col_out.text().strip(),
            out_w=self.cW.value(), out_h=self.cH.value(),
            images_per_collage=self.cK.value(), collage_count=self.cN.value(),
            min_scale=float(self.minS.value()), max_scale=float(self.maxS.value()),
            max_rotation_deg=float(self.rot.value()),
            allow_reuse_if_insufficient=self.reuse.isChecked(),
            margin=self.margin.value(),
            background_mode=self.bg_mode.currentText(),
            bg_color_hex=self.bg_hex.text(),
            add_drop_shadow=self.shadow.isChecked(),
            use_bg_removal_if_no_alpha=self.rembg.isChecked(),
            placement=self.place.currentText(),
            seed=seed
        )
        self.collage_worker.log.connect(self._log)
        self.collage_worker.finished.connect(lambda: self._log("=== Collage done ==="))
        self.collage_worker.start()

    def stop_collage(self):
        if self.collage_worker:
            self.collage_worker.stop()
            self.collage_worker.wait()
            self.collage_worker = None
            self._log("=== Collage stopped ===")

    # ---------- Pipeline Orchestration ----------
    def run_pipeline(self):
        # lock UI bits if you want; for simplicity we just chain steps via signals
        kws = [p.strip() for p in re.split(r"[,;\n]+", self.pipe_keywords.text().strip()) if p.strip()]
        if not kws:
            self._log("!! Enter pipeline keywords.")
            return

        # step 1: scrape
        self._log("=== Pipeline: Scrape starting ===")
        self.scrape_worker = ImageScraperWorker(
            keywords=kws,
            target=self.pipe_target.value(),
            out_dir=self.pipe_scrape_out.text().strip(),
            provider=self.pipe_provider.currentText(),
            headless=self.pipe_headless.isChecked()
        )
        self.scrape_worker.log.connect(self._log)
        self.scrape_worker.finished.connect(self._pipeline_after_scrape)
        self.scrape_worker.start()

    def _pipeline_after_scrape(self):
        self._log("=== Pipeline: Scrape finished → Extract starting ===")
        # step 2: extract
        self.extract_worker = ExtractWorker(
            input_dir=self.pipe_extract_in.text().strip() or self.pipe_scrape_out.text().strip(),
            output_dir=self.pipe_extract_out.text().strip(),
            out_w=self.pipe_ex_w.value(), out_h=self.pipe_ex_h.value()
        )
        self.extract_worker.log.connect(self._log)
        self.extract_worker.finished.connect(self._pipeline_after_extract)
        self.extract_worker.start()

    def _pipeline_after_extract(self):
        self._log("=== Pipeline: Extract finished → Collage starting ===")
        # step 3: collage
        seed = None
        if self.pipe_seed.text().strip():
            try: seed = int(self.pipe_seed.text().strip())
            except ValueError: self._log("!! Seed must be int (ignored).")
        self.collage_worker = CollageWorker(
            input_dir=self.pipe_col_in.text().strip() or self.pipe_extract_out.text().strip(),
            output_dir=self.pipe_col_out.text().strip(),
            out_w=self.pipe_W.value(), out_h=self.pipe_H.value(),
            images_per_collage=self.pipe_K.value(), collage_count=self.pipe_N.value(),
            min_scale=float(self.pipe_minS.value()), max_scale=float(self.pipe_maxS.value()),
            max_rotation_deg=float(self.pipe_rot.value()),
            allow_reuse_if_insufficient=self.pipe_reuse.isChecked(),
            margin=self.pipe_margin.value(),
            background_mode=self.pipe_bg_mode.currentText(),
            bg_color_hex=self.pipe_bg_hex.text(),
            add_drop_shadow=self.pipe_shadow.isChecked(),
            use_bg_removal_if_no_alpha=self.pipe_rembg.isChecked(),
            placement=self.pipe_place.currentText(),
            seed=seed
        )
        self.collage_worker.log.connect(self._log)
        self.collage_worker.finished.connect(lambda: self._log("=== Pipeline complete ==="))
        self.collage_worker.start()

    # ---------- helpers ----------
    def _log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _choose_dir_into(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Choose folder", line_edit.text())
        if path:
            line_edit.setText(path)

    def _hbox(self, widgets: list[QWidget]) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w); l.setContentsMargins(0,0,0,0)
        for wid in widgets: l.addWidget(wid)
        return w

    # Clean shutdown
    def closeEvent(self, event):
        for worker in (self.scrape_worker, self.extract_worker, self.collage_worker):
            if worker and worker.isRunning():
                self._log("Stopping worker before exit…")
                worker.stop()
                worker.wait()
        event.accept()


# ===========================
# Run
# ===========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PipelineGUI()
    gui.show()
    sys.exit(app.exec())
