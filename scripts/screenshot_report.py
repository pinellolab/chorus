"""Take a screenshot of an HTML variant report using headless Chrome.

Usage:
  mamba run -n chorus python scripts/screenshot_report.py path/to/report.html [output.png]
"""
import sys
import os
import time


def screenshot_html(html_path: str, output_path: str | None = None, width: int = 1400, wait_seconds: int = 5):
    """Render an HTML file in headless Chrome and save a full-page screenshot."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    if output_path is None:
        output_path = html_path.replace('.html', '_screenshot.png')

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'--window-size={width},2000')

    driver = webdriver.Chrome(options=options)

    try:
        file_url = f'file://{os.path.abspath(html_path)}'
        driver.get(file_url)

        # Wait for IGV.js to render
        time.sleep(wait_seconds)

        # Get full page height
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(width, total_height + 200)
        time.sleep(1)

        driver.save_screenshot(output_path)
        print(f'Screenshot saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)')
    finally:
        driver.quit()

    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <report.html> [output.png]')
        sys.exit(1)

    html = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    screenshot_html(html, out)
