# nftmax_pro.py
"""
NFTMax Pro - Advanced Image Art Pipeline
Enhanced version with improved robustness, features, and modern UI
"""

import os
import re
import sys
import math
import json
import time
import random
import threading
import traceback
import subprocess
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

# ---------- Third-party ----------
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QGridLayout, QTabWidget, QMessageBox, QGroupBox, QProgressBar,
    QSlider, QFrame, QScrollArea, QSplitter
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QIcon

from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

import requests
from urllib.parse import urlparse, parse_qs, unquote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService

try:
    import undetected_chromedriver as uc
    HAS_UC = True
except Exception:
    HAS_UC = False

try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WDM = True
except Exception:
    HAS_WDM = False


# ===========================
# Constants & Configuration
# ===========================

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

class BlendMode(Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"

class FilterEffect(Enum):
    NONE = "none"
    BLUR = "blur"
    SHARPEN = "sharpen"
    EDGE_ENHANCE = "edge_enhance"
    VINTAGE = "vintage"
    VIGNETTE = "vignette"


@dataclass
class CollageConfig:
    """Configuration for collage generation"""
    canvas_size: Tuple[int, int]
    images_per_collage: int
    collage_count: int
    min_scale: float
    max_scale: float
    max_rotation: float
    margin: int
    placement: str
    background_mode: str
    bg_color: str
    add_shadow: bool
    shadow_blur: int
    shadow_opacity: int
    shadow_offset: Tuple[int, int]
    allow_reuse: bool
    use_bg_removal: bool
    blend_mode: str
    filter_effect: str
    border_enabled: bool
    border_width: int
    border_color: str
    seed: Optional[int]
    brightness_variance: float
    contrast_variance: float
    saturation_variance: float


# ===========================
# Utility Functions
# ===========================

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name).strip("_ ").replace(" ", "_")[:200]

def list_images_recursive(root: Path) -> List[Path]:
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]

def has_meaningful_alpha(img: Image.Image, threshold: int = 5) -> bool:
    if img.mode != "RGBA":
        return False
    alpha = img.split()[-1]
    if alpha.getbbox() is None:
        return False
    w, h = alpha.size
    samples, nonfull = 0, 0
    step_x, step_y = max(1, w // 64), max(1, h // 64)
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

def apply_filter(img: Image.Image, filter_name: str) -> Image.Image:
    """Apply various filter effects to images"""
    if filter_name == FilterEffect.BLUR.value:
        return img.filter(ImageFilter.GaussianBlur(2))
    elif filter_name == FilterEffect.SHARPEN.value:
        return img.filter(ImageFilter.SHARPEN)
    elif filter_name == FilterEffect.EDGE_ENHANCE.value:
        return img.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_name == FilterEffect.VINTAGE.value:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.7)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.2)
    elif filter_name == FilterEffect.VIGNETTE.value:
        return apply_vignette(img)
    return img

def apply_vignette(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """Apply vignette effect"""
    width, height = img.size
    vignette = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(vignette)
    
    for i in range(min(width, height) // 2):
        alpha = int(255 * (1 - intensity * (1 - i / (min(width, height) / 2))))
        draw.ellipse([i, i, width - i, height - i], fill=alpha)
    
    vignette = vignette.filter(ImageFilter.GaussianBlur(50))
    result = img.copy()
    result.putalpha(Image.composite(img.split()[-1] if img.mode == 'RGBA' else Image.new('L', img.size, 255), 
                                    Image.new('L', img.size, 0), vignette))
    return result

def add_border(img: Image.Image, width: int, color: str) -> Image.Image:
    """Add border around image"""
    if width <= 0:
        return img
    
    hex_color = color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    try:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    except:
        r, g, b = 255, 255, 255
    
    bordered = Image.new("RGBA", (img.width + width*2, img.height + width*2), (r, g, b, 255))
    bordered.paste(img, (width, width), img if img.mode == 'RGBA' else None)
    return bordered

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
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    return img.resize((nw, nh), Image.LANCZOS)

def place_random_position(canvas_size: Tuple[int, int], img_size: Tuple[int, int], 
                          margin: int) -> Tuple[int, int]:
    W, H = canvas_size
    w, h = img_size
    x = random.randint(-margin, max(-margin, W - w + margin))
    y = random.randint(-margin, max(-margin, H - h + margin))
    return x, y

def add_shadow(overlay: Image.Image, blur_radius: int = 10, opacity: int = 140, 
               offset: Tuple[int, int] = (8, 8)) -> Image.Image:
    ox, oy = offset
    w, h = overlay.size
    shadow = Image.new("RGBA", (w + abs(ox) * 2, h + abs(oy) * 2), (0, 0, 0, 0))
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
    combined.alpha_composite(overlay, (abs(ox), abs(oy)))
    return combined

def adjust_image_properties(img: Image.Image, brightness: float = 1.0, 
                           contrast: float = 1.0, saturation: float = 1.0) -> Image.Image:
    """Adjust brightness, contrast, and saturation"""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)
    return img

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
# Chrome Detection
# ===========================

def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)

def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout.strip()
    except Exception:
        return 1, ""

def detect_chrome_binary() -> Optional[str]:
    env = os.getenv("CHROME_BINARY")
    if env and Path(env).exists():
        return env

    mac_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        str(Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    for p in mac_paths:
        if Path(p).exists():
            return p

    for c in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
        p = _which(c)
        if p:
            return p

    win_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles%\Chromium\Application\chrome.exe"),
    ]
    for p in win_paths:
        if Path(p).exists():
            return p

    return None

def detect_chrome_major_version(binary: Optional[str]) -> Optional[int]:
    out = ""
    if binary:
        code, out = _run([binary, "--version"])
        if code == 0 and out:
            pass
    if not out:
        for c in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
            code, out = _run([c, "--version"])
            if code == 0 and out:
                break
    if not out:
        for p in [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]:
            if Path(p).exists():
                code, out = _run([p, "--version"])
                if code == 0 and out:
                    break

    m = re.search(r"(\d+)\.(\d+)\.(\d+)\.(\d+)", out or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# ===========================
# Enhanced Scraper Worker
# ===========================

class ImageScraperWorker(QThread):
    log = Signal(str)
    finished = Signal()
    progress = Signal(str, int, int)
    status_update = Signal(str)

    def __init__(self, keywords: List[str], target: int, out_dir: str,
                 provider: str = "Auto", headless: bool = True, 
                 min_width: int = 0, min_height: int = 0):
        super().__init__()
        self.keywords = [k.strip() for k in keywords if k.strip()]
        self.target = target
        self.out_dir = out_dir
        self.provider = provider
        self.headless = headless
        self.min_width = min_width
        self.min_height = min_height
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _make_driver(self):
        binary = detect_chrome_binary()
        major = detect_chrome_major_version(binary)
        headless_flag = "--headless=new" if self.headless else None

        self.log.emit(f"[Driver] Chrome: {binary or 'not found'}")
        self.log.emit(f"[Driver] Version: {major or 'unknown'}")

        if HAS_UC:
            try:
                self.status_update.emit("Initializing undetected Chrome driver...")
                uc_opts = uc.ChromeOptions()
                if headless_flag:
                    uc_opts.add_argument(headless_flag)
                uc_opts.add_argument(f"--user-agent={USER_AGENT}")
                uc_opts.add_argument("--disable-gpu")
                uc_opts.add_argument("--no-sandbox")
                uc_opts.add_argument("--disable-dev-shm-usage")
                uc_opts.add_argument("--window-size=1920,1080")
                if binary:
                    uc_opts.binary_location = binary
                driver = uc.Chrome(version_main=major, options=uc_opts) if major else uc.Chrome(options=uc_opts)
                driver.set_page_load_timeout(60)
                self.log.emit("[Driver] Using undetected_chromedriver ✅")
                return driver
            except Exception as e:
                self.log.emit(f"[Driver] UC failed: {e}")

        try:
            self.status_update.emit("Trying Selenium Manager...")
            chrome_opts = Options()
            if headless_flag:
                chrome_opts.add_argument(headless_flag)
            chrome_opts.add_argument(f"--user-agent={USER_AGENT}")
            chrome_opts.add_argument("--disable-gpu")
            chrome_opts.add_argument("--no-sandbox")
            chrome_opts.add_argument("--disable-dev-shm-usage")
            chrome_opts.add_argument("--window-size=1920,1080")
            if binary:
                chrome_opts.binary_location = binary
            driver = webdriver.Chrome(options=chrome_opts)
            driver.set_page_load_timeout(60)
            self.log.emit("[Driver] Using Selenium Manager ✅")
            return driver
        except Exception as e:
            self.log.emit(f"[Driver] Selenium Manager failed: {e}")

        if HAS_WDM:
            try:
                self.status_update.emit("Trying webdriver-manager...")
                chrome_opts = Options()
                if headless_flag:
                    chrome_opts.add_argument(headless_flag)
                chrome_opts.add_argument(f"--user-agent={USER_AGENT}")
                chrome_opts.add_argument("--disable-gpu")
                chrome_opts.add_argument("--no-sandbox")
                chrome_opts.add_argument("--disable-dev-shm-usage")
                chrome_opts.add_argument("--window-size=1920,1080")
                if binary:
                    chrome_opts.binary_location = binary
                service = ChromeService(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_opts)
                driver.set_page_load_timeout(60)
                self.log.emit("[Driver] Using webdriver-manager ✅")
                return driver
            except Exception as e:
                self.log.emit(f"[Driver] webdriver-manager failed: {e}")

        raise RuntimeError("Unable to create Chrome WebDriver")

    def _filter_by_size(self, url: str, session: requests.Session) -> bool:
        """Check if image meets minimum size requirements"""
        if self.min_width == 0 and self.min_height == 0:
            return True
        
        try:
            response = session.head(url, timeout=5)
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) < 1024:  # Skip very small files
                return False
        except:
            pass
        
        return True

    def _bing_urls(self, driver, keyword: str, target: int) -> List[str]:
        search_url = f"https://www.bing.com/images/search?q={requests.utils.quote(keyword)}&form=HDRSC2&first=1"
        self.status_update.emit(f"Searching Bing for: {keyword}")
        driver.get(search_url)
        time.sleep(2)
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
            time.sleep(1.2)
            before = len(urls)
            harvest()
            self.progress.emit(keyword, len(urls), target)
            stagnant = stagnant + 1 if len(urls) == before else 0
            if stagnant > 6:
                break
        return urls[:target]

    def _ddg_urls(self, driver, keyword: str, target: int) -> List[str]:
        url = f"https://duckduckgo.com/?q={requests.utils.quote(keyword)}&ia=images&iar=images"
        self.status_update.emit(f"Searching DuckDuckGo for: {keyword}")
        driver.get(url)
        time.sleep(2)
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
            time.sleep(1.2)
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
            self.log.emit("!! Bing low results, trying DuckDuckGo...")
            urls.extend(self._ddg_urls(driver, keyword, target - len(urls)))
        return urls[:target]

    def run(self):
        ensure_dir(self.out_dir)
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        
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
                self.log.emit(f"\n=== Scraping: {kw} ===")
                
                urls = self._collect(driver, kw, self.target)
                self.log.emit(f"Found {len(urls)} URLs")
                
                count = 0
                for idx, u in enumerate(urls, 1):
                    if self._stop.is_set(): break
                    
                    base = os.path.join(out_folder, f"{sanitize_filename(kw)}_{idx:04d}")
                    try:
                        r = session.get(u, timeout=25, stream=True)
                        r.raise_for_status()
                        
                        ext = guess_extension(u, r.headers.get("Content-Type", ""))
                        dest = base + ext
                        
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                        
                        # Validate image
                        try:
                            img = Image.open(dest)
                            if self.min_width > 0 and img.width < self.min_width:
                                os.remove(dest)
                                continue
                            if self.min_height > 0 and img.height < self.min_height:
                                os.remove(dest)
                                continue
                        except:
                            if os.path.exists(dest):
                                os.remove(dest)
                            continue
                        
                        count += 1
                        self.log.emit(f"✔ [{count}/{len(urls)}] {Path(dest).name}")
                        self.status_update.emit(f"Downloaded {count}/{len(urls)} images for {kw}")
                        
                    except Exception as e:
                        self.log.emit(f"✗ Failed: {e}")
                    
                    time.sleep(0.2)
                
                self.log.emit(f"Completed {kw}: {count} images saved")
                
        finally:
            try:
                driver.quit()
            except:
                pass
            self.status_update.emit("Scraping complete")
            self.finished.emit()


# ===========================
# Enhanced Extractor Worker
# ===========================

class ExtractWorker(QThread):
    log = Signal(str)
    finished = Signal()
    progress = Signal(int, int)
    status_update = Signal(str)

    def __init__(self, input_dir, output_dir, out_w, out_h, 
                 maintain_aspect: bool = True, quality: int = 95):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.out_w = out_w
        self.out_h = out_h
        self.maintain_aspect = maintain_aspect
        self.quality = quality
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        if not self.input_dir.exists():
            self.log.emit("!! Input folder not found")
            self.finished.emit()
            return

        images = list_images_recursive(self.input_dir)
        self.log.emit(f"Found {len(images)} images to process")
        self.status_update.emit(f"Processing {len(images)} images...")

        for idx, img_path in enumerate(images, 1):
            if self._stop: break
            
            try:
                self.progress.emit(idx, len(images))
                self.log.emit(f"[{idx}/{len(images)}] {img_path.name}")
                
                with open(img_path, "rb") as f:
                    input_bytes = f.read()

                if not HAS_REMBG:
                    img = Image.open(BytesIO(input_bytes)).convert("RGBA")
                    out_img = img
                else:
                    out_bytes = rembg_remove(input_bytes)
                    out_img = Image.open(BytesIO(out_bytes)).convert("RGBA")

                if self.maintain_aspect:
                    out_img.thumbnail((self.out_w, self.out_h), Image.LANCZOS)
                else:
                    out_img = out_img.resize((self.out_w, self.out_h), Image.LANCZOS)

                rel = img_path.relative_to(self.input_dir)
                out_file = self.output_dir / rel.with_suffix(".png")
                out_file.parent.mkdir(parents=True, exist_ok=True)
                
                out_img.save(out_file, "PNG", optimize=True)
                self.log.emit(f"✔ Saved: {out_file.name}")
                self.status_update.emit(f"Processed {idx}/{len(images)}")
                
            except Exception as e:
                self.log.emit(f"✗ Failed {img_path.name}: {e}")

        self.status_update.emit("Extraction complete")
        self.finished.emit()


# ===========================
# Enhanced Collage Worker
# ===========================

class CollageWorker(QThread):
    log = Signal(str)
    finished = Signal()
    progress = Signal(int, int)
    status_update = Signal(str)

    def __init__(self, config: CollageConfig, input_dir: str, output_dir: str):
        super().__init__()
        self.config = config
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._stop = False

    def stop(self):
        self._stop = True

    def _make_canvas(self) -> Image.Image:
        W, H = self.config.canvas_size
        if self.config.background_mode == "Transparent":
            return Image.new("RGBA", (W, H), (0, 0, 0, 0))
        
        hexv = self.config.bg_color.strip().lstrip("#")
        if len(hexv) == 3:
            hexv = "".join([c*2 for c in hexv])
        try:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
        except:
            r, g, b = 10, 10, 10
        return Image.new("RGBA", (W, H), (r, g, b, 255))

    def _choose_positions(self, count: int) -> List[Tuple[int, int]]:
        W, H = self.config.canvas_size
        positions: List[Tuple[int, int]] = []
        
        if self.config.placement == "Grid-ish":
            cols = max(1, int(math.sqrt(count)))
            rows = max(1, math.ceil(count / cols))
            cell_w = max(1, W // cols)
            cell_h = max(1, H // rows)
            
            for r in range(rows):
                for c in range(cols):
                    if len(positions) >= count:
                        break
                    cx = c * cell_w + cell_w // 2
                    cy = r * cell_h + cell_h // 2
                    jitter_x = random.randint(-cell_w // 4, cell_w // 4)
                    jitter_y = random.randint(-cell_h // 4, cell_h // 4)
                    x = max(-self.config.margin, min(W - 1, cx + jitter_x))
                    y = max(-self.config.margin, min(H - 1, cy + jitter_y))
                    positions.append((x, y))
            random.shuffle(positions)
            return positions
            
        elif self.config.placement == "Center Bias":
            for _ in range(count):
                mu_x, mu_y = W / 2, H / 2
                sigma_x, sigma_y = W / 6, H / 6
                x = int(random.gauss(mu_x, sigma_x))
                y = int(random.gauss(mu_y, sigma_y))
                positions.append((x, y))
            random.shuffle(positions)
            return positions
            
        elif self.config.placement == "Circular":
            center_x, center_y = W // 2, H // 2
            radius = min(W, H) // 3
            for i in range(count):
                angle = 2 * math.pi * i / count
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                positions.append((x, y))
            return positions
            
        else:  # Random
            for _ in range(count):
                positions.append(place_random_position(self.config.canvas_size, (0, 0), self.config.margin))
            return positions

    def run(self):
        if self.config.seed is not None:
            random.seed(self.config.seed)

        if not self.input_dir.exists():
            self.log.emit("!! Collage input directory not found")
            self.finished.emit()
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_imgs = list_images_recursive(self.input_dir)
        
        if not all_imgs:
            self.log.emit("!! No images found for collage")
            self.finished.emit()
            return

        self.log.emit(f"Found {len(all_imgs)} source images")
        self.status_update.emit(f"Creating {self.config.collage_count} collages...")

        required_unique = self.config.images_per_collage * self.config.collage_count
        if len(all_imgs) < required_unique and not self.config.allow_reuse:
            max_collages = len(all_imgs) // max(1, self.config.images_per_collage)
            self.log.emit(f"!! Adjusted collage count to {max_collages} (insufficient unique images)")
            self.config = CollageConfig(
                canvas_size=self.config.canvas_size,
                images_per_collage=self.config.images_per_collage,
                collage_count=max_collages,
                min_scale=self.config.min_scale,
                max_scale=self.config.max_scale,
                max_rotation=self.config.max_rotation,
                margin=self.config.margin,
                placement=self.config.placement,
                background_mode=self.config.background_mode,
                bg_color=self.config.bg_color,
                add_shadow=self.config.add_shadow,
                shadow_blur=self.config.shadow_blur,
                shadow_opacity=self.config.shadow_opacity,
                shadow_offset=self.config.shadow_offset,
                allow_reuse=self.config.allow_reuse,
                use_bg_removal=self.config.use_bg_removal,
                blend_mode=self.config.blend_mode,
                filter_effect=self.config.filter_effect,
                border_enabled=self.config.border_enabled,
                border_width=self.config.border_width,
                border_color=self.config.border_color,
                seed=self.config.seed,
                brightness_variance=self.config.brightness_variance,
                contrast_variance=self.config.contrast_variance,
                saturation_variance=self.config.saturation_variance
            )

        random.shuffle(all_imgs)
        used: Set[Path] = set()
        produced = 0

        for ci in range(self.config.collage_count):
            if self._stop:
                break
                
            self.progress.emit(ci + 1, self.config.collage_count)
            self.status_update.emit(f"Creating collage {ci + 1}/{self.config.collage_count}")
            
            canvas = self._make_canvas()
            chosen: List[Path] = []
            attempts = 0
            
            while len(chosen) < self.config.images_per_collage and attempts < 10000:
                attempts += 1
                remaining = [p for p in all_imgs if (self.config.allow_reuse or p not in used)]
                if not remaining:
                    break
                p = random.choice(remaining)
                if not self.config.allow_reuse and p in used:
                    continue
                chosen.append(p)
                if not self.config.allow_reuse:
                    used.add(p)

            if not chosen:
                self.log.emit(f"✗ Skip collage {ci + 1}: no images available")
                continue

            positions = self._choose_positions(len(chosen))

            for i, img_path in enumerate(chosen):
                if self._stop:
                    break
                    
                try:
                    img = Image.open(img_path).convert("RGBA")
                except Exception as e:
                    self.log.emit(f"✗ Failed to open {img_path.name}: {e}")
                    continue

                # Background removal if needed
                if self.config.use_bg_removal and not has_meaningful_alpha(img):
                    img = auto_remove_bg(img)

                # Scale to canvas
                img = random_scale_to_canvas(
                    img, 
                    self.config.canvas_size,
                    self.config.min_scale,
                    self.config.max_scale
                )

                # Random property adjustments
                if self.config.brightness_variance > 0:
                    brightness = 1.0 + random.uniform(-self.config.brightness_variance, self.config.brightness_variance)
                else:
                    brightness = 1.0
                    
                if self.config.contrast_variance > 0:
                    contrast = 1.0 + random.uniform(-self.config.contrast_variance, self.config.contrast_variance)
                else:
                    contrast = 1.0
                    
                if self.config.saturation_variance > 0:
                    saturation = 1.0 + random.uniform(-self.config.saturation_variance, self.config.saturation_variance)
                else:
                    saturation = 1.0

                img = adjust_image_properties(img, brightness, contrast, saturation)

                # Apply filter
                if self.config.filter_effect != FilterEffect.NONE.value:
                    img = apply_filter(img, self.config.filter_effect)

                # Add border
                if self.config.border_enabled and self.config.border_width > 0:
                    img = add_border(img, self.config.border_width, self.config.border_color)

                # Rotation
                if self.config.max_rotation > 0:
                    angle = random.uniform(-self.config.max_rotation, self.config.max_rotation)
                    img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

                # Shadow
                if self.config.add_shadow:
                    img = add_shadow(
                        img,
                        blur_radius=self.config.shadow_blur,
                        opacity=self.config.shadow_opacity,
                        offset=self.config.shadow_offset
                    )

                # Position
                if self.config.placement == "Random":
                    xy = place_random_position(self.config.canvas_size, img.size, self.config.margin)
                else:
                    if positions:
                        cx, cy = positions.pop(0)
                    else:
                        cx, cy = place_random_position(self.config.canvas_size, img.size, self.config.margin)
                    xy = (int(cx - img.width / 2), int(cy - img.height / 2))

                # Composite onto canvas
                canvas.alpha_composite(img, dest=xy)

            # Save collage
            out_path = self.output_dir / f"collage_{ci + 1:04d}.png"
            try:
                canvas.save(out_path, "PNG", optimize=True)
                produced += 1
                self.log.emit(f"✔ Saved: {out_path.name}")
            except Exception as e:
                self.log.emit(f"✗ Failed to save collage {ci + 1}: {e}")

        self.log.emit(f"\n=== Complete: {produced} collages created ===")
        self.status_update.emit(f"Complete: {produced} collages created")
        self.finished.emit()


# ===========================
# Enhanced Modern GUI
# ===========================

class ModernLineEdit(QLineEdit):
    """Styled line edit with modern appearance"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
            }
        """)


class ModernButton(QPushButton):
    """Styled button with modern appearance"""
    def __init__(self, text, primary=False):
        super().__init__(text)
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #1084d8;
                }
                QPushButton:pressed {
                    background-color: #006cbd;
                }
                QPushButton:disabled {
                    background-color: #3a3a3a;
                    color: #6a6a6a;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d2d;
                    color: #e0e0e0;
                    border: 1px solid #3a3a3a;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                    border: 1px solid #0078d4;
                }
                QPushButton:pressed {
                    background-color: #252525;
                }
            """)


class NFTMaxProGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NFTMax Pro - Advanced Image Art Pipeline")
        self.setGeometry(100, 100, 1400, 900)
        
        self.scrape_worker: Optional[ImageScraperWorker] = None
        self.extract_worker: Optional[ExtractWorker] = None
        self.collage_worker: Optional[CollageWorker] = None
        
        self._setup_theme()
        self._build_ui()

    def _setup_theme(self):
        """Apply dark modern theme"""
        self.setStyleSheet("""
            QWidget {
                background-color: #0f0f0f;
                color: #e0e0e0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #2d2d2d;
                background-color: #151515;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #1e1e1e;
                color: #a0a0a0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #151515;
                color: #0078d4;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #252525;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #2d2d2d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                background-color: #0078d4;
                color: white;
                border-radius: 3px;
            }
            QLabel {
                color: #c0c0c0;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 5px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #0078d4;
            }
            QComboBox {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox:focus {
                border: 1px solid #0078d4;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #e0e0e0;
                selection-background-color: #0078d4;
            }
            QCheckBox {
                spacing: 8px;
                color: #c0c0c0;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #3a3a3a;
                border-radius: 3px;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #0d0d0d;
                color: #00ff88;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                padding: 8px;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #3a3a3a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4a4a4a;
            }
        """)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)
        
        # Header
        header = QLabel("NFTMax Pro")
        header.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #0078d4;
                padding: 10px;
            }
        """)
        root.addWidget(header)
        
        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                color: #00ff88;
                padding: 8px 12px;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        root.addWidget(self.status_bar)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)
        
        # Splitter for tabs and logs
        splitter = QSplitter(Qt.Vertical)
        
        # Tabs
        self.tabs = QTabWidget()
        self._build_scraper_tab()
        self._build_extractor_tab()
        self._build_collage_tab()
        self._build_pipeline_tab()
        
        splitter.addWidget(self.tabs)
        
        # Log area
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_header = QHBoxLayout()
        log_label = QLabel("📋 Console Output")
        log_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        log_header.addWidget(log_label)
        
        clear_btn = ModernButton("Clear", primary=False)
        clear_btn.clicked.connect(lambda: self.log_box.clear())
        log_header.addWidget(clear_btn)
        log_header.addStretch()
        
        log_layout.addLayout(log_header)
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(200)
        log_layout.addWidget(self.log_box)
        
        splitter.addWidget(log_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        root.addWidget(splitter, 1)

    def _build_scraper_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Configuration group
        config_group = QGroupBox("Scraper Configuration")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(12)
        
        row = 0
        config_layout.addWidget(QLabel("Keywords (comma-separated):"), row, 0)
        self.scr_kw = QTextEdit()
        self.scr_kw.setPlaceholderText("cat, rabbit, solana logo, abstract art")
        self.scr_kw.setMaximumHeight(80)
        self.scr_kw.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0; border: 1px solid #3a3a3a; border-radius: 4px;")
        config_layout.addWidget(self.scr_kw, row, 1, 1, 3)
        row += 1
        
        config_layout.addWidget(QLabel("Images per keyword:"), row, 0)
        self.scr_target = QSpinBox()
        self.scr_target.setRange(1, 1000)
        self.scr_target.setValue(30)
        config_layout.addWidget(self.scr_target, row, 1)
        
        config_layout.addWidget(QLabel("Provider:"), row, 2)
        self.scr_provider = QComboBox()
        self.scr_provider.addItems(["Auto", "Bing", "DuckDuckGo"])
        config_layout.addWidget(self.scr_provider, row, 3)
        row += 1
        
        config_layout.addWidget(QLabel("Min width:"), row, 0)
        self.scr_min_w = QSpinBox()
        self.scr_min_w.setRange(0, 8192)
        self.scr_min_w.setValue(256)
        config_layout.addWidget(self.scr_min_w, row, 1)
        
        config_layout.addWidget(QLabel("Min height:"), row, 2)
        self.scr_min_h = QSpinBox()
        self.scr_min_h.setRange(0, 8192)
        self.scr_min_h.setValue(256)
        config_layout.addWidget(self.scr_min_h, row, 3)
        row += 1
        
        self.scr_headless = QCheckBox("Headless mode")
        self.scr_headless.setChecked(True)
        config_layout.addWidget(self.scr_headless, row, 0, 1, 2)
        row += 1
        
        config_layout.addWidget(QLabel("Output directory:"), row, 0)
        self.scr_out = ModernLineEdit(str(Path.cwd() / "scraped_images"))
        config_layout.addWidget(self.scr_out, row, 1, 1, 2)
        btn_out = ModernButton("Browse...")
        btn_out.clicked.connect(lambda: self._choose_dir(self.scr_out))
        config_layout.addWidget(btn_out, row, 3)
        
        layout.addWidget(config_group)
        
        # Controls
        controls = QHBoxLayout()
        controls.addStretch()
        self.scr_start_btn = ModernButton("Start Scraping", primary=True)
        self.scr_start_btn.clicked.connect(self.start_scrape)
        controls.addWidget(self.scr_start_btn)
        
        self.scr_stop_btn = ModernButton("Stop")
        self.scr_stop_btn.clicked.connect(self.stop_scrape)
        self.scr_stop_btn.setEnabled(False)
        controls.addWidget(self.scr_stop_btn)
        
        layout.addLayout(controls)
        layout.addStretch()
        
        self.tabs.addTab(tab, "🔍 Scraper")

    def _build_extractor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        config_group = QGroupBox("Extractor Configuration")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(12)
        
        row = 0
        config_layout.addWidget(QLabel("Input directory:"), row, 0)
        self.ex_in = ModernLineEdit(str(Path.cwd() / "scraped_images"))
        config_layout.addWidget(self.ex_in, row, 1, 1, 2)
        btn_in = ModernButton("Browse...")
        btn_in.clicked.connect(lambda: self._choose_dir(self.ex_in))
        config_layout.addWidget(btn_in, row, 3)
        row += 1
        
        config_layout.addWidget(QLabel("Output directory:"), row, 0)
        self.ex_out = ModernLineEdit(str(Path.cwd() / "extracted"))
        config_layout.addWidget(self.ex_out, row, 1, 1, 2)
        btn_out = ModernButton("Browse...")
        btn_out.clicked.connect(lambda: self._choose_dir(self.ex_out))
        config_layout.addWidget(btn_out, row, 3)
        row += 1
        
        config_layout.addWidget(QLabel("Output width:"), row, 0)
        self.ex_w = QSpinBox()
        self.ex_w.setRange(16, 4096)
        self.ex_w.setValue(512)
        config_layout.addWidget(self.ex_w, row, 1)
        
        config_layout.addWidget(QLabel("Output height:"), row, 2)
        self.ex_h = QSpinBox()
        self.ex_h.setRange(16, 4096)
        self.ex_h.setValue(512)
        config_layout.addWidget(self.ex_h, row, 3)
        row += 1
        
        self.ex_maintain_aspect = QCheckBox("Maintain aspect ratio")
        self.ex_maintain_aspect.setChecked(True)
        config_layout.addWidget(self.ex_maintain_aspect, row, 0, 1, 2)
        row += 1
        
        layout.addWidget(config_group)
        
        # Controls
        controls = QHBoxLayout()
        controls.addStretch()
        self.ex_start_btn = ModernButton("Start Extraction", primary=True)
        self.ex_start_btn.clicked.connect(self.start_extract)
        controls.addWidget(self.ex_start_btn)
        
        self.ex_stop_btn = ModernButton("Stop")
        self.ex_stop_btn.clicked.connect(self.stop_extract)
        self.ex_stop_btn.setEnabled(False)
        controls.addWidget(self.ex_stop_btn)
        
        layout.addLayout(controls)
        layout.addStretch()
        
        self.tabs.addTab(tab, "✂️ Extractor")

    def _build_collage_tab(self):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Basic settings
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QGridLayout(basic_group)
        basic_layout.setSpacing(12)
        
        row = 0
        basic_layout.addWidget(QLabel("Input directory:"), row, 0)
        self.col_in = ModernLineEdit(str(Path.cwd() / "extracted"))
        basic_layout.addWidget(self.col_in, row, 1, 1, 2)
        btn_in = ModernButton("Browse...")
        btn_in.clicked.connect(lambda: self._choose_dir(self.col_in))
        basic_layout.addWidget(btn_in, row, 3)
        row += 1
        
        basic_layout.addWidget(QLabel("Output directory:"), row, 0)
        self.col_out = ModernLineEdit(str(Path.cwd() / "collages"))
        basic_layout.addWidget(self.col_out, row, 1, 1, 2)
        btn_out = ModernButton("Browse...")
        btn_out.clicked.connect(lambda: self._choose_dir(self.col_out))
        basic_layout.addWidget(btn_out, row, 3)
        row += 1
        
        basic_layout.addWidget(QLabel("Canvas width:"), row, 0)
        self.col_w = QSpinBox()
        self.col_w.setRange(128, 8192)
        self.col_w.setValue(1920)
        basic_layout.addWidget(self.col_w, row, 1)
        
        basic_layout.addWidget(QLabel("Canvas height:"), row, 2)
        self.col_h = QSpinBox()
        self.col_h.setRange(128, 8192)
        self.col_h.setValue(1080)
        basic_layout.addWidget(self.col_h, row, 3)
        row += 1
        
        basic_layout.addWidget(QLabel("Images per collage:"), row, 0)
        self.col_k = QSpinBox()
        self.col_k.setRange(1, 200)
        self.col_k.setValue(18)
        basic_layout.addWidget(self.col_k, row, 1)
        
        basic_layout.addWidget(QLabel("Number of collages:"), row, 2)
        self.col_n = QSpinBox()
        self.col_n.setRange(1, 10000)
        self.col_n.setValue(10)
        basic_layout.addWidget(self.col_n, row, 3)
        row += 1
        
        layout.addWidget(basic_group)
        
        # Layout settings
        layout_group = QGroupBox("Layout & Positioning")
        layout_grid = QGridLayout(layout_group)
        layout_grid.setSpacing(12)
        
        row = 0
        layout_grid.addWidget(QLabel("Placement mode:"), row, 0)
        self.col_place = QComboBox()
        self.col_place.addItems(["Random", "Center Bias", "Grid-ish", "Circular"])
        layout_grid.addWidget(self.col_place, row, 1)
        
        layout_grid.addWidget(QLabel("Margin:"), row, 2)
        self.col_margin = QSpinBox()
        self.col_margin.setRange(0, 2000)
        self.col_margin.setValue(60)
        layout_grid.addWidget(self.col_margin, row, 3)
        row += 1
        
        layout_grid.addWidget(QLabel("Min scale:"), row, 0)
        self.col_min_scale = QDoubleSpinBox()
        self.col_min_scale.setRange(0.02, 2.0)
        self.col_min_scale.setSingleStep(0.01)
        self.col_min_scale.setValue(0.18)
        layout_grid.addWidget(self.col_min_scale, row, 1)
        
        layout_grid.addWidget(QLabel("Max scale:"), row, 2)
        self.col_max_scale = QDoubleSpinBox()
        self.col_max_scale.setRange(0.02, 3.0)
        self.col_max_scale.setSingleStep(0.01)
        self.col_max_scale.setValue(0.55)
        layout_grid.addWidget(self.col_max_scale, row, 3)
        row += 1
        
        layout_grid.addWidget(QLabel("Max rotation (deg):"), row, 0)
        self.col_rot = QDoubleSpinBox()
        self.col_rot.setRange(0.0, 180.0)
        self.col_rot.setSingleStep(1.0)
        self.col_rot.setValue(28.0)
        layout_grid.addWidget(self.col_rot, row, 1)
        row += 1
        
        layout.addWidget(layout_group)
        
        # Appearance settings
        appearance_group = QGroupBox("Appearance & Effects")
        appearance_grid = QGridLayout(appearance_group)
        appearance_grid.setSpacing(12)
        
        row = 0
        appearance_grid.addWidget(QLabel("Background mode:"), row, 0)
        self.col_bg_mode = QComboBox()
        self.col_bg_mode.addItems(["Transparent", "Solid"])
        appearance_grid.addWidget(self.col_bg_mode, row, 1)
        
        appearance_grid.addWidget(QLabel("Background color:"), row, 2)
        self.col_bg_hex = ModernLineEdit("#0a0a0a")
        appearance_grid.addWidget(self.col_bg_hex, row, 3)
        row += 1
        
        appearance_grid.addWidget(QLabel("Filter effect:"), row, 0)
        self.col_filter = QComboBox()
        self.col_filter.addItems([e.value for e in FilterEffect])
        appearance_grid.addWidget(self.col_filter, row, 1)
        
        appearance_grid.addWidget(QLabel("Blend mode:"), row, 2)
        self.col_blend = QComboBox()
        self.col_blend.addItems([b.value for b in BlendMode])
        appearance_grid.addWidget(self.col_blend, row, 3)
        row += 1
        
        self.col_shadow = QCheckBox("Drop shadow")
        self.col_shadow.setChecked(True)
        appearance_grid.addWidget(self.col_shadow, row, 0)
        
        appearance_grid.addWidget(QLabel("Shadow blur:"), row, 1)
        self.col_shadow_blur = QSpinBox()
        self.col_shadow_blur.setRange(0, 50)
        self.col_shadow_blur.setValue(10)
        appearance_grid.addWidget(self.col_shadow_blur, row, 2)
        
        appearance_grid.addWidget(QLabel("Shadow opacity:"), row, 3)
        self.col_shadow_opacity = QSpinBox()
        self.col_shadow_opacity.setRange(0, 255)
        self.col_shadow_opacity.setValue(140)
        appearance_grid.addWidget(self.col_shadow_opacity, row, 4)
        row += 1
        
        self.col_border = QCheckBox("Add border")
        self.col_border.setChecked(False)
        appearance_grid.addWidget(self.col_border, row, 0)
        
        appearance_grid.addWidget(QLabel("Border width:"), row, 1)
        self.col_border_width = QSpinBox()
        self.col_border_width.setRange(0, 50)
        self.col_border_width.setValue(5)
        appearance_grid.addWidget(self.col_border_width, row, 2)
        
        appearance_grid.addWidget(QLabel("Border color:"), row, 3)
        self.col_border_color = ModernLineEdit("#ffffff")
        appearance_grid.addWidget(self.col_border_color, row, 4)
        row += 1
        
        layout.addWidget(appearance_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_grid = QGridLayout(advanced_group)
        advanced_grid.setSpacing(12)
        
        row = 0
        advanced_grid.addWidget(QLabel("Brightness variance:"), row, 0)
        self.col_brightness = QDoubleSpinBox()
        self.col_brightness.setRange(0.0, 1.0)
        self.col_brightness.setSingleStep(0.05)
        self.col_brightness.setValue(0.1)
        advanced_grid.addWidget(self.col_brightness, row, 1)
        
        advanced_grid.addWidget(QLabel("Contrast variance:"), row, 2)
        self.col_contrast = QDoubleSpinBox()
        self.col_contrast.setRange(0.0, 1.0)
        self.col_contrast.setSingleStep(0.05)
        self.col_contrast.setValue(0.1)
        advanced_grid.addWidget(self.col_contrast, row, 3)
        row += 1
        
        advanced_grid.addWidget(QLabel("Saturation variance:"), row, 0)
        self.col_saturation = QDoubleSpinBox()
        self.col_saturation.setRange(0.0, 1.0)
        self.col_saturation.setSingleStep(0.05)
        self.col_saturation.setValue(0.15)
        advanced_grid.addWidget(self.col_saturation, row, 1)
        
        advanced_grid.addWidget(QLabel("Random seed:"), row, 2)
        self.col_seed = ModernLineEdit()
        self.col_seed.setPlaceholderText("Optional (integer)")
        advanced_grid.addWidget(self.col_seed, row, 3)
        row += 1
        
        self.col_reuse = QCheckBox("Allow image reuse if insufficient")
        self.col_reuse.setChecked(False)
        advanced_grid.addWidget(self.col_reuse, row, 0, 1, 2)
        
        self.col_bg_remove = QCheckBox("Auto background removal")
        self.col_bg_remove.setChecked(False)
        if not HAS_REMBG:
            self.col_bg_remove.setText("Auto BG removal (install rembg)")
            self.col_bg_remove.setEnabled(False)
        advanced_grid.addWidget(self.col_bg_remove, row, 2, 1, 2)
        row += 1
        
        layout.addWidget(advanced_group)
        
        # Controls
        controls = QHBoxLayout()
        controls.addStretch()
        self.col_start_btn = ModernButton("Start Creating Collages", primary=True)
        self.col_start_btn.clicked.connect(self.start_collage)
        controls.addWidget(self.col_start_btn)
        
        self.col_stop_btn = ModernButton("Stop")
        self.col_stop_btn.clicked.connect(self.stop_collage)
        self.col_stop_btn.setEnabled(False)
        controls.addWidget(self.col_stop_btn)
        
        layout.addLayout(controls)
        layout.addStretch()
        
        self.tabs.addTab(scroll, "🎨 Collage Maker")

    def _build_pipeline_tab(self):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        info_label = QLabel("⚡ Run the complete pipeline: Scrape → Extract → Collage")
        info_label.setStyleSheet("""
            QLabel {
                background-color: #1e3a5f;
                color: #ffffff;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(info_label)
        
        # Quick config
        quick_group = QGroupBox("Quick Pipeline Setup")
        quick_layout = QGridLayout(quick_group)
        quick_layout.setSpacing(12)
        
        row = 0
        quick_layout.addWidget(QLabel("Keywords:"), row, 0)
        self.pipe_kw = QTextEdit()
        self.pipe_kw.setPlaceholderText("cat, rabbit, abstract art")
        self.pipe_kw.setMaximumHeight(60)
        self.pipe_kw.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0; border: 1px solid #3a3a3a; border-radius: 4px;")
        quick_layout.addWidget(self.pipe_kw, row, 1, 1, 3)
        row += 1
        
        quick_layout.addWidget(QLabel("Images per keyword:"), row, 0)
        self.pipe_target = QSpinBox()
        self.pipe_target.setRange(1, 500)
        self.pipe_target.setValue(25)
        quick_layout.addWidget(self.pipe_target, row, 1)
        
        quick_layout.addWidget(QLabel("Collages to create:"), row, 2)
        self.pipe_collages = QSpinBox()
        self.pipe_collages.setRange(1, 1000)
        self.pipe_collages.setValue(10)
        quick_layout.addWidget(self.pipe_collages, row, 3)
        row += 1
        
        quick_layout.addWidget(QLabel("Working directory:"), row, 0)
        self.pipe_work_dir = ModernLineEdit(str(Path.cwd() / "nftmax_output"))
        quick_layout.addWidget(self.pipe_work_dir, row, 1, 1, 2)
        btn_work = ModernButton("Browse...")
        btn_work.clicked.connect(lambda: self._choose_dir(self.pipe_work_dir))
        quick_layout.addWidget(btn_work, row, 3)
        row += 1
        
        self.pipe_headless = QCheckBox("Headless scraping")
        self.pipe_headless.setChecked(True)
        quick_layout.addWidget(self.pipe_headless, row, 0, 1, 2)
        row += 1
        
        layout.addWidget(quick_group)
        
        # Pipeline status
        status_group = QGroupBox("Pipeline Status")
        status_layout = QVBoxLayout(status_group)
        
        self.pipe_status_label = QLabel("Not started")
        self.pipe_status_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #00ff88;
                padding: 8px;
            }
        """)
        status_layout.addWidget(self.pipe_status_label)
        
        self.pipe_progress = QProgressBar()
        self.pipe_progress.setVisible(False)
        status_layout.addWidget(self.pipe_progress)
        
        layout.addWidget(status_group)
        
        # Controls
        controls = QHBoxLayout()
        controls.addStretch()
        
        self.pipe_start_btn = ModernButton("🚀 Run Full Pipeline", primary=True)
        self.pipe_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa55;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #00cc66;
            }
            QPushButton:pressed {
                background-color: #008844;
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
                color: #6a6a6a;
            }
        """)
        self.pipe_start_btn.clicked.connect(self.run_pipeline)
        controls.addWidget(self.pipe_start_btn)
        
        self.pipe_stop_btn = ModernButton("Stop Pipeline")
        self.pipe_stop_btn.clicked.connect(self.stop_pipeline)
        self.pipe_stop_btn.setEnabled(False)
        controls.addWidget(self.pipe_stop_btn)
        
        layout.addLayout(controls)
        layout.addStretch()
        
        self.tabs.addTab(scroll, "⚡ Full Pipeline")

    # ==================== Worker Management ====================
    
    def start_scrape(self):
        text = self.scr_kw.toPlainText().strip()
        if not text:
            self._log("❌ Please enter keywords")
            return
            
        keywords = [k.strip() for k in re.split(r"[,;\n]+", text) if k.strip()]
        
        self.scrape_worker = ImageScraperWorker(
            keywords=keywords,
            target=self.scr_target.value(),
            out_dir=self.scr_out.text().strip(),
            provider=self.scr_provider.currentText(),
            headless=self.scr_headless.isChecked(),
            min_width=self.scr_min_w.value(),
            min_height=self.scr_min_h.value()
        )
        
        self.scrape_worker.log.connect(self._log)
        self.scrape_worker.status_update.connect(self._update_status)
        self.scrape_worker.finished.connect(self._on_scrape_finished)
        
        self.scr_start_btn.setEnabled(False)
        self.scr_stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self._log("🔍 Starting scraper...")
        self.scrape_worker.start()

    def stop_scrape(self):
        if self.scrape_worker:
            self._log("⏸️ Stopping scraper...")
            self.scrape_worker.stop()
            self.scrape_worker.wait()
            self._on_scrape_finished()

    def _on_scrape_finished(self):
        self.scr_start_btn.setEnabled(True)
        self.scr_stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._log("✅ Scraping complete")
        self._update_status("Ready")

    def start_extract(self):
        in_dir = self.ex_in.text().strip()
        out_dir = self.ex_out.text().strip()
        
        if not in_dir or not out_dir:
            self._log("❌ Please select input and output directories")
            return
            
        self.extract_worker = ExtractWorker(
            input_dir=in_dir,
            output_dir=out_dir,
            out_w=self.ex_w.value(),
            out_h=self.ex_h.value(),
            maintain_aspect=self.ex_maintain_aspect.isChecked()
        )
        
        self.extract_worker.log.connect(self._log)
        self.extract_worker.status_update.connect(self._update_status)
        self.extract_worker.progress.connect(self._update_progress)
        self.extract_worker.finished.connect(self._on_extract_finished)
        
        self.ex_start_btn.setEnabled(False)
        self.ex_stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        
        self._log("✂️ Starting extraction...")
        self.extract_worker.start()

    def stop_extract(self):
        if self.extract_worker:
            self._log("⏸️ Stopping extractor...")
            self.extract_worker.stop()
            self.extract_worker.wait()
            self._on_extract_finished()

    def _on_extract_finished(self):
        self.ex_start_btn.setEnabled(True)
        self.ex_stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._log("✅ Extraction complete")
        self._update_status("Ready")

    def start_collage(self):
        seed = None
        if self.col_seed.text().strip():
            try:
                seed = int(self.col_seed.text().strip())
            except ValueError:
                self._log("⚠️ Invalid seed (must be integer), ignoring...")
        
        config = CollageConfig(
            canvas_size=(self.col_w.value(), self.col_h.value()),
            images_per_collage=self.col_k.value(),
            collage_count=self.col_n.value(),
            min_scale=self.col_min_scale.value(),
            max_scale=self.col_max_scale.value(),
            max_rotation=self.col_rot.value(),
            margin=self.col_margin.value(),
            placement=self.col_place.currentText(),
            background_mode=self.col_bg_mode.currentText(),
            bg_color=self.col_bg_hex.text(),
            add_shadow=self.col_shadow.isChecked(),
            shadow_blur=self.col_shadow_blur.value(),
            shadow_opacity=self.col_shadow_opacity.value(),
            shadow_offset=(8, 8),
            allow_reuse=self.col_reuse.isChecked(),
            use_bg_removal=self.col_bg_remove.isChecked(),
            blend_mode=self.col_blend.currentText(),
            filter_effect=self.col_filter.currentText(),
            border_enabled=self.col_border.isChecked(),
            border_width=self.col_border_width.value(),
            border_color=self.col_border_color.text(),
            seed=seed,
            brightness_variance=self.col_brightness.value(),
            contrast_variance=self.col_contrast.value(),
            saturation_variance=self.col_saturation.value()
        )
        
        self.collage_worker = CollageWorker(
            config=config,
            input_dir=self.col_in.text().strip(),
            output_dir=self.col_out.text().strip()
        )
        
        self.collage_worker.log.connect(self._log)
        self.collage_worker.status_update.connect(self._update_status)
        self.collage_worker.progress.connect(self._update_progress)
        self.collage_worker.finished.connect(self._on_collage_finished)
        
        self.col_start_btn.setEnabled(False)
        self.col_stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        
        self._log("🎨 Starting collage creation...")
        self.collage_worker.start()

    def stop_collage(self):
        if self.collage_worker:
            self._log("⏸️ Stopping collage maker...")
            self.collage_worker.stop()
            self.collage_worker.wait()
            self._on_collage_finished()

    def _on_collage_finished(self):
        self.col_start_btn.setEnabled(True)
        self.col_stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._log("✅ Collage creation complete")
        self._update_status("Ready")

    def run_pipeline(self):
        text = self.pipe_kw.toPlainText().strip()
        if not text:
            self._log("❌ Please enter keywords for pipeline")
            return
        
        keywords = [k.strip() for k in re.split(r"[,;\n]+", text) if k.strip()]
        work_dir = Path(self.pipe_work_dir.text().strip())
        
        self.pipe_start_btn.setEnabled(False)
        self.pipe_stop_btn.setEnabled(True)
        self.pipe_progress.setVisible(True)
        self.pipe_progress.setRange(0, 3)
        self.pipe_progress.setValue(0)
        
        # Stage 1: Scrape
        self._log("\n" + "="*60)
        self._log("🚀 PIPELINE STARTED")
        self._log("="*60)
        self._log("📋 Stage 1/3: Scraping images...")
        self.pipe_status_label.setText("Stage 1/3: Scraping...")
        
        scrape_dir = work_dir / "1_scraped"
        
        self.scrape_worker = ImageScraperWorker(
            keywords=keywords,
            target=self.pipe_target.value(),
            out_dir=str(scrape_dir),
            provider="Auto",
            headless=self.pipe_headless.isChecked(),
            min_width=256,
            min_height=256
        )
        
        self.scrape_worker.log.connect(self._log)
        self.scrape_worker.status_update.connect(self._update_status)
        self.scrape_worker.finished.connect(lambda: self._pipeline_stage_extract(work_dir))
        self.scrape_worker.start()

    def _pipeline_stage_extract(self, work_dir: Path):
        self.pipe_progress.setValue(1)
        self._log("\n📋 Stage 2/3: Extracting and processing...")
        self.pipe_status_label.setText("Stage 2/3: Extracting...")
        
        scrape_dir = work_dir / "1_scraped"
        extract_dir = work_dir / "2_extracted"
        
        self.extract_worker = ExtractWorker(
            input_dir=str(scrape_dir),
            output_dir=str(extract_dir),
            out_w=512,
            out_h=512,
            maintain_aspect=True
        )
        
        self.extract_worker.log.connect(self._log)
        self.extract_worker.status_update.connect(self._update_status)
        self.extract_worker.finished.connect(lambda: self._pipeline_stage_collage(work_dir))
        self.extract_worker.start()

    def _pipeline_stage_collage(self, work_dir: Path):
        self.pipe_progress.setValue(2)
        self._log("\n📋 Stage 3/3: Creating collages...")
        self.pipe_status_label.setText("Stage 3/3: Creating collages...")
        
        extract_dir = work_dir / "2_extracted"
        collage_dir = work_dir / "3_collages"
        
        config = CollageConfig(
            canvas_size=(1920, 1080),
            images_per_collage=18,
            collage_count=self.pipe_collages.value(),
            min_scale=0.18,
            max_scale=0.55,
            max_rotation=28.0,
            margin=60,
            placement="Random",
            background_mode="Transparent",
            bg_color="#0a0a0a",
            add_shadow=True,
            shadow_blur=10,
            shadow_opacity=140,
            shadow_offset=(8, 8),
            allow_reuse=False,
            use_bg_removal=False,
            blend_mode="normal",
            filter_effect="none",
            border_enabled=False,
            border_width=5,
            border_color="#ffffff",
            seed=None,
            brightness_variance=0.1,
            contrast_variance=0.1,
            saturation_variance=0.15
        )
        
        self.collage_worker = CollageWorker(
            config=config,
            input_dir=str(extract_dir),
            output_dir=str(collage_dir)
        )
        
        self.collage_worker.log.connect(self._log)
        self.collage_worker.status_update.connect(self._update_status)
        self.collage_worker.finished.connect(self._on_pipeline_finished)
        self.collage_worker.start()

    def _on_pipeline_finished(self):
        self.pipe_progress.setValue(3)
        self.pipe_start_btn.setEnabled(True)
        self.pipe_stop_btn.setEnabled(False)
        self.pipe_status_label.setText("✅ Pipeline Complete!")
        self._log("\n" + "="*60)
        self._log("✅ PIPELINE COMPLETE!")
        self._log("="*60)
        self._update_status("Pipeline complete")

    def stop_pipeline(self):
        self._log("⏸️ Stopping pipeline...")
        if self.scrape_worker:
            self.scrape_worker.stop()
        if self.extract_worker:
            self.extract_worker.stop()
        if self.collage_worker:
            self.collage_worker.stop()
        
        self.pipe_start_btn.setEnabled(True)
        self.pipe_stop_btn.setEnabled(False)
        self.pipe_progress.setVisible(False)
        self.pipe_status_label.setText("Stopped")
        self._update_status("Ready")

    # ==================== Utility Methods ====================
    
    def _log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def _update_status(self, status: str):
        self.status_bar.setText(f"Status: {status}")

    def _update_progress(self, current: int, total: int):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _choose_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text()
        )
        if path:
            line_edit.setText(path)

    def closeEvent(self, event):
        """Clean shutdown"""
        workers = [self.scrape_worker, self.extract_worker, self.collage_worker]
        for worker in workers:
            if worker and worker.isRunning():
                self._log("Stopping worker before exit...")
                worker.stop()
                worker.wait()
        event.accept()


# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application metadata
    app.setApplicationName("NFTMax Pro")
    app.setOrganizationName("NFTMax")
    
    gui = NFTMaxProGUI()
    gui.show()
    
    sys.exit(app.exec())
