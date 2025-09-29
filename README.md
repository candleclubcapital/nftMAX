A powerful **end-to-end image art generation pipeline** with a sleek **PySide6 GUI**.
Scrape → Extract → Collage — all in one application.

This tool allows you to:

* Scrape images from the web (Bing / DuckDuckGo)
* Remove backgrounds automatically (via [rembg](https://github.com/danielgatis/rembg))
* Composite images into randomized or structured collages with artistic controls

---

## ✨ Features

* **GUI** built with **PySide6** — intuitive tabbed interface
* **Image Scraper** (Bing, DuckDuckGo, fallback logic, headless Chrome/UC support)
* **Background Extraction** with [rembg] (optional, falls back gracefully)
* **Collage Generator**:

  * Transparent or solid background
  * Random, grid-like, or center-biased placement
  * Random scaling, rotation, drop-shadows
  * Seeded randomness for reproducible runs
  * Supports thousands of collages in batch
* **Pipeline Runner**: chain Scrape → Extract → Collage in one click
* **Logging Console**: real-time logs inside the GUI

---

## 🖥️ Screenshots

*(Add your own screenshots here — GUI with tabs and generated collages would look great!)*

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourname/art-pipeline-gui.git
cd art-pipeline-gui
pip install -r requirements.txt
```

### Requirements

* Python 3.10+
* [Google Chrome](https://www.google.com/chrome/) installed
* **Dependencies**:

  * `PySide6`
  * `Pillow`
  * `selenium`
  * `requests`
  * `undetected-chromedriver` (optional, improves scraping reliability)
  * `rembg` (optional, for background removal)

Install with:

```bash
pip install PySide6 Pillow selenium requests
pip install rembg undetected-chromedriver
```

---

## 🚀 Usage

Run the app:

```bash
python art_pipeline_gui.py
```

The GUI provides 3 tabs:

1. **Scrape**

   * Enter keywords → scrape images from Bing/DDG → save locally
   * Example: `"cat, rabbit, solana logo"`

2. **Extract**

   * Removes backgrounds from scraped images
   * Resizes to your chosen W×H
   * Outputs `.png` with alpha

3. **Collage**

   * Combine extracted images into collages
   * Configure:

     * canvas size, number of images per collage
     * scale range, max rotation, margin
     * background (solid/transparent)
     * drop shadow / rembg fallback
   * Generate batches (`N` collages)

4. **Pipeline Runner**

   * Run **Scrape → Extract → Collage** in one go

---

## ⚙️ Example Workflow

```text
Keywords: cat, rabbit
Target per keyword: 25
Scrape output: ./scraped_images

Extract size: 512 × 512
Extract output: ./extracted

Collage canvas: 1920 × 1080
Images per collage: 18
Number of collages: 10
Placement: Grid-ish
Background: Solid #0a0a0a
Drop shadows: enabled
```

Generates **10 collages** with randomized arrangements of cats and rabbits 🌌

---

## 🛠 Development Notes

* Scraper uses Selenium + Chrome.
* If scraping fails:

  * Check Chrome installation
  * Try `undetected-chromedriver` for bypassing blocks
* Background removal is optional — install `rembg` for best results.

---

## 📂 Project Structure

```
art_pipeline_gui.py     # main application
scraped_images/         # default scrape output
extracted/              # extracted images
collages_out/           # final collages
```

---

## 🤝 Contributing

Pull requests are welcome!
Ideas for improvements:

* Add more image providers (Google, Yandex, etc.)
* GPU-accelerated compositing
* Custom layer templates
* ML-based collage arrangement




