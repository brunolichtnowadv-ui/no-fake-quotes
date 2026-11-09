#!/usr/bin/env python3
"""
quote_checker.py - check that every quotation in a draft exists, word for word, in a saved source.

Built to catch the classic AI-writing failure: a quotation that looks right but does not exist,
or that was silently changed. It does not judge relevance or validity, only literal correspondence.

Usage:
  python3 quote_checker.py draft.docx sources/             folder with the sources (PDF, TXT, MD, HTML, DOCX)
  python3 quote_checker.py draft.md a.pdf b.txt            individual sources; the draft can be .docx, .md or .txt
  python3 quote_checker.py draft.docx sources/ --ocr       OCR (Tesseract) on PDF pages with no text layer
  python3 quote_checker.py draft.docx sources/ --min 6     minimum words per fragment (default 5)

What counts as a quotation:
  (1) in .docx, a block paragraph indented at least --indent-cm from the left (default 3.5 cm);
  (2) in .md, a block quote (lines starting with ">");
  (3) anywhere, text between quotation marks with at least --min-quote words (default 8).

Before searching, the checker removes a final reference in parentheses (e.g. a case citation after
a quoted ruling) and markers such as "[emphasis added]" or "[grifou-se]", then splits the quote at
ellipses ("[...]", "(...)", "...") and checks each fragment on its own. Comparison is word by word,
ignoring accents, case and punctuation, and tolerating end-of-line hyphenation.

Output: one line per quotation.
  OK          found, with file and page
  DIVERGE     a close match exists (similarity %), with the words that differ, so the quote can be fixed
  NOT FOUND   no source contains it: find and save the source, or drop the quotation
Exit code: 0 all OK; 1 at least one DIVERGE or NOT FOUND; 2 execution failure.
Requires only the Python standard library plus `pdftotext` (poppler-utils) for PDFs;
`--ocr` also needs `pdftoppm` and `tesseract`.
"""
import argparse
import difflib
import html
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
QUOTE_RE = re.compile(r'[“"«]([^”"»]+)[”"»]')
MARKERS_RE = re.compile(
    r'\[(emphasis added|emphasis mine|italics added|sic|grifou-se|grifo nosso|grifos nossos|destacou-se|g\.n\.)\]',
    re.I)


def norm(s):
    """Lower case, no accents, no punctuation, single spaces; joins end-of-line hyphenation."""
    s = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', s)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r'[^0-9a-z]+', ' ', s)
    return ' '.join(s.split())


# ---------- draft

def _indent(p):
    ind = p.find(W + 'pPr/' + W + 'ind')
    if ind is None:
        return 0
    v = ind.get(W + 'left') or ind.get(W + 'start') or '0'
    try:
        return int(v)
    except ValueError:
        return 0


def _quoted(txt, n, min_quote, out):
    for m in QUOTE_RE.finditer(txt):
        if len(m.group(1).split()) >= min_quote:
            out.append(('quote', n, m.group(1)))


def quotes_from_draft(path, indent_cm, min_quote):
    path = Path(path)
    out = []
    if path.suffix.lower() == '.docx':
        min_twips = int(indent_cm * 567)
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read('word/document.xml'))
        n = 0
        for p in root.iter(W + 'p'):
            txt = ''.join(t.text or '' for t in p.iter(W + 't'))
            if not txt.strip():
                continue
            n += 1
            if _indent(p) >= min_twips:
                out.append(('block', n, txt))
            else:
                _quoted(txt, n, min_quote, out)
        return out
    # .md / .txt: paragraphs separated by blank lines; ">" lines form block quotes
    text = path.read_text(encoding='utf-8', errors='ignore')
    for n, para in enumerate((p for p in re.split(r'\n\s*\n', text) if p.strip()), 1):
        lines = para.splitlines()
        if all(l.lstrip().startswith('>') for l in lines):
            out.append(('block', n, ' '.join(l.lstrip()[1:].strip() for l in lines)))
        else:
            _quoted(para, n, min_quote, out)
    return out


def fragments(txt, minimum):
    t = MARKERS_RE.sub(' ', txt)
    t = re.sub(r'\s*\((?:[^()]|\([^()]*\))*\)\s*\.?\s*$', '', t.strip())  # final reference in parentheses
    parts = re.split(r'\[\s*(?:\.\.\.|…)\s*\]|\(\s*(?:\.\.\.|…)\s*\)|…|\.\.\.', t)
    out = []
    for part in parts:
        nn = norm(part)
        if len(nn.split()) >= minimum:
            out.append((part.strip(), nn))
    return out


# ---------- sources

def pdf_pages(path, ocr, lang):
    with tempfile.TemporaryDirectory() as d:
        txt = Path(d) / 'x.txt'
        subprocess.run(['pdftotext', '-layout', str(path), str(txt)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pages = txt.read_text(encoding='utf-8', errors='ignore').split('\f')
        if pages and not pages[-1].strip():
            pages.pop()  # pdftotext ends with a form feed
        if ocr and shutil.which('tesseract') and shutil.which('pdftoppm'):
            for i, tx in enumerate(pages):
                if len(tx.strip()) < 40:
                    base = Path(d) / ('p%d' % (i + 1))
                    subprocess.run(['pdftoppm', '-r', '300', '-f', str(i + 1), '-l', str(i + 1),
                                    '-png', '-singlefile', str(path), str(base)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    png = base.with_suffix('.png')
                    if png.exists():
                        r = subprocess.run(['tesseract', str(png), '-', '-l', lang],
                                           capture_output=True, text=True)
                        pages[i] = r.stdout
    return pages


def load_sources(items, ocr, lang):
    files = []
    for it in items:
        p = Path(it)
        files += sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p]
    sources, no_text = [], []
    for f in files:
        ext = f.suffix.lower()
        try:
            if ext == '.pdf':
                pages = pdf_pages(f, ocr, lang)
            elif ext in ('.txt', '.md'):
                pages = [f.read_text(encoding='utf-8', errors='ignore')]
            elif ext in ('.html', '.htm'):
                raw = f.read_text(encoding='utf-8', errors='ignore')
                raw = re.sub(r'(?is)<(script|style).*?</\1>', ' ', raw)
                pages = [html.unescape(re.sub(r'<[^>]+>', ' ', raw))]
            elif ext == '.docx':
                with zipfile.ZipFile(f) as z:
                    r = ET.fromstring(z.read('word/document.xml'))
                pages = ['\n'.join(''.join(t.text or '' for t in p.iter(W + 't')) for p in r.iter(W + 'p'))]
            else:
                continue
        except Exception as e:  # noqa
            print('WARNING unreadable source: %s (%s)' % (f.name, e))
            continue
        for i, tx in enumerate(pages, 1):
            nn = norm(tx)
            if nn:
                sources.append((f.name, i if ext == '.pdf' else None, nn))
            elif ext == '.pdf':
                no_text.append('%s p.%d' % (f.name, i))
    return sources, no_text


# ---------- search

def locate(frag, sources):
    for name, page, tx in sources:
        if frag in tx:
            return ('OK', name, page, 100, '')
    # quotation that crosses a page break: join neighbouring pages of the same file
    for k in range(len(sources) - 1):
        if sources[k][0] == sources[k + 1][0] and frag in sources[k][2] + ' ' + sources[k + 1][2]:
            return ('OK', sources[k][0], '%s-%s' % (sources[k][1], sources[k + 1][1]), 100, '')
    # closest match: same-length word window, anchored on the longest words of the fragment
    fw = frag.split()
    best = (0, None, None, '')
    anchors = sorted(set(fw), key=len, reverse=True)[:3]
    for name, page, tx in sources:
        tw = tx.split()
        for i in [i for i, w in enumerate(tw) if w in anchors][:400]:
            start = max(0, i - fw.index(tw[i]))
            window = tw[start:start + len(fw)]
            r = difflib.SequenceMatcher(None, fw, window).ratio()
            if r > best[0]:
                best = (r, name, page, ' '.join(window))
    r, name, page, window = best
    if r >= 0.75:
        return ('DIVERGE', name, page, round(r * 100), window)
    return ('NOT', name, page, round(r * 100), '')


def where(name, page):
    return name + (' p.%s' % page if page else '')


def difference(a, b):
    sa, sb = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb).get_opcodes():
        if op != 'equal':
            out.append('draft "%s" vs source "%s"' % (' '.join(sa[i1:i2]) or '-', ' '.join(sb[j1:j2]) or '-'))
    return '; '.join(out[:3])


def main():
    ap = argparse.ArgumentParser(description='Check every quotation in a draft against the saved sources.')
    ap.add_argument('draft', help='.docx, .md or .txt')
    ap.add_argument('sources', nargs='+', help='source files or folders (PDF, TXT, MD, HTML, DOCX)')
    ap.add_argument('--ocr', action='store_true', help='OCR PDF pages that have no text layer')
    ap.add_argument('--lang', default='eng', help='Tesseract language(s) for --ocr, e.g. eng, por, eng+por')
    ap.add_argument('--min', type=int, default=5, help='minimum words per fragment (default 5)')
    ap.add_argument('--min-quote', type=int, default=8, help='minimum words for an inline quote (default 8)')
    ap.add_argument('--indent-cm', type=float, default=3.5, help='left indent that marks a block quote in .docx')
    a = ap.parse_args()
    try:
        quotes = quotes_from_draft(a.draft, a.indent_cm, a.min_quote)
        sources, no_text = load_sources(a.sources, a.ocr, a.lang)
    except Exception as e:  # noqa
        print('FAILURE: %s' % e)
        return 2
    if not sources:
        print('ERROR no source with text: save the sources (PDF, ruling, statute, article) and run again.')
        return 1
    problems = 0
    for kind, n, txt in quotes:
        frs = fragments(txt, a.min)
        if not frs:
            continue
        res = [(f, locate(nn, sources)) for f, nn in frs]
        label = '#%d %s "%s"' % (n, kind, txt.strip()[:60] + ('...' if len(txt.strip()) > 60 else ''))
        bad = [(f, r) for f, r in res if r[0] != 'OK']
        if not bad:
            print('OK        %s -> %s' % (label, ', '.join(sorted({where(r[1], r[2]) for _, r in res}))))
            continue
        problems += 1
        for f, r in bad:
            if r[0] == 'DIVERGE':
                print('DIVERGE   %s -> %s (%d%%): %s' % (label, where(r[1], r[2]), r[3], difference(norm(f), r[4])))
            else:
                print('NOT FOUND %s -> fragment "%s"' % (label, f[:80]))
    if no_text:
        print('WARNING pages with no text (use --ocr or check the image): %s' % ', '.join(no_text[:10]))
    print('Summary: %d quotations, %d with problems, %d source pages read.' % (len(quotes), problems, len(sources)))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
