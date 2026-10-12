import os
import sys
import subprocess
from pathlib import Path
from markdown_it import MarkdownIt

def md_to_html_and_pdf(md_file_path: str, output_pdf_path: str):
    md_path = Path(md_file_path).resolve()
    pdf_path = Path(output_pdf_path).resolve()
    html_path = md_path.with_suffix(".html")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    md = MarkdownIt("commonmark", {"html": True, "breaks": True})
    html_body = md.render(md_text)
    
    # Styled HTML with print CSS
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Project Submission Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  
  @page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
  }}
  
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1f2937;
    background-color: #ffffff;
    line-height: 1.6;
    font-size: 10.5pt;
    margin: 0;
    padding: 0;
  }}

  h1, h2, h3, h4 {{
    color: #111827;
    font-weight: 600;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
  }}

  h1 {{
    font-size: 18pt;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.3em;
    margin-top: 0;
  }}

  h2 {{
    font-size: 14pt;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.2em;
  }}

  h3 {{
    font-size: 12pt;
  }}

  p, li {{
    margin: 0.5em 0;
  }}

  code {{
    font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
    font-size: 9pt;
    background-color: #f3f4f6;
    padding: 2px 4px;
    border-radius: 4px;
    color: #0f172a;
  }}

  pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 1em 0;
  }}

  pre code {{
    background: transparent;
    padding: 0;
    font-size: 8.5pt;
    line-height: 1.45;
  }}

  blockquote {{
    border-left: 4px solid #3b82f6;
    background-color: #eff6ff;
    padding: 8px 16px;
    margin: 1em 0;
    border-radius: 0 4px 4px 0;
    color: #1e40af;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1.5em 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }}

  th, td {{
    border: 1px solid #e5e7eb;
    padding: 8px 12px;
    text-align: left;
  }}

  th {{
    background-color: #f9fafb;
    font-weight: 600;
  }}

  tr:nth-child(even) {{
    background-color: #fcfdfd;
  }}

  img {{
    max-width: 90%;
    height: auto;
    display: block;
    margin: 1.5em auto;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    page-break-inside: avoid;
  }}

  hr {{
    border: 0;
    height: 1px;
    background: #e5e7eb;
    margin: 2em 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"HTML saved to: {html_path}")
    
    # Run Headless Chrome to generate PDF
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome_path).exists():
        print("Chrome not found. HTML file generated.")
        return
        
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        str(html_path)
    ]
    
    print(f"Generating PDF with Chrome headless...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully generated PDF: {pdf_path}")
    else:
        print(f"Error running Chrome: {result.stderr}")

if __name__ == "__main__":
    md_file = sys.argv[1] if len(sys.argv) > 1 else "project_submission_report.md"
    out_pdf = sys.argv[2] if len(sys.argv) > 2 else "project_submission_report.pdf"
    md_to_html_and_pdf(md_file, out_pdf)
