#!/usr/bin/env python3
"""
highlight.py - pixel-accurate highlighter for excerpts cut from PDFs, scans and screenshots.

(English summary; the code comments and command-line options are in Portuguese.)
The highlight is drawn from the REAL position of each letter, never estimated: character boxes
from the PDF text layer, or word boxes from OCR for images and scans. After cropping, it checks the
result in the pixels with an independent OCR pass: no partly highlighted word, no highlight over
blank space, no highlight bleeding into the line above or below. It also writes a "magnifier" image
with both ends of each highlighted line enlarged, for a quick visual check.

Examples:
  python3 highlight.py --pdf "Document.pdf" --pagina 3 --trecho "literal passage" --saida "excerpt.png"
  python3 highlight.py --imagem "screenshot.png" --trecho "literal passage" --saida "excerpt.png"
  python3 highlight.py --conferir "excerpt.png" --lupa "magnifier.png"
Options: --pagina = page, --trecho = passage (repeatable), --saida = output, --ocorrencia = occurrence,
--linhas-antes/--linhas-depois = context lines, --recorte = crop box in PDF points or "pagina",
--cor = colour (default FFFF00), --conferir = check an existing excerpt, --lupa = magnifier output.
Exit code: 0 no errors; 1 passage not found or check failed; 2 execution failure.
Requires pdfplumber (PDF mode), Pillow and numpy; the pixel check and image mode need Tesseract.

---- Original description (Portuguese) ----
Destaques (marca-texto) perfeitamente precisos nas colagens.

O destaque é calculado a partir da posição REAL de cada letra, nunca por estimativa:
  - PDF com camada de texto: caixas dos caracteres extraídas do próprio PDF;
  - imagem, print ou PDF digitalizado: caixas das palavras obtidas por OCR (Tesseract).
Depois de gerar o recorte, o script confere o resultado nos pixels (OCR independente):
nenhuma palavra parcialmente destacada, nenhuma sobra de destaque onde não há texto,
nenhum destaque invadindo a linha de cima ou de baixo e nenhuma interrupção suspeita
na quebra de linha. Gera também a "lupa": as pontas de cada linha destacada ampliadas,
para a conferência visual obrigatória.

Uso:
  python3 highlight.py --pdf "Documento.pdf" --pagina 3 --trecho "texto literal" --saida "recorte.png"
        [--trecho "outro trecho"] [--ocorrencia 2] [--linhas-antes 1] [--linhas-depois 1]
        [--recorte x0,topo,x1,base | --recorte pagina] [--dpi 220] [--cor FFFF00] [--ocr]
  python3 highlight.py --imagem "print.png" --trecho "texto literal" --saida "recorte.png"
  python3 highlight.py --conferir "recorte.png" [--lupa "lupa.png"]

Coordenadas de --recorte em pontos do PDF (origem no canto superior esquerdo da página).
Código de saída: 0 sem erros; 1 trecho não localizado ou conferência com erros; 2 falha de execução.
Requer pdfplumber (modo PDF), Pillow e numpy; a conferência nos pixels e o modo imagem
exigem o Tesseract (apt-get install -y tesseract-ocr tesseract-ocr-por; pip install pytesseract).
"""
import argparse
import difflib
import os
import re
import sys
import unicodedata

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - outros scripts podem importar este módulo de forma opcional
    np = None
    Image = None

# ---------------------------------------------------------------- normalização do texto

EQUIV = {}
for _c in '“”„‟«»″"':
    EQUIV[_c] = '"'
for _c in "‘’‚‛′`´'":
    EQUIV[_c] = "'"
for _c in '‐‑‒–—―−':
    EQUIV[_c] = '-'
HIFENS = set('-‐‑­')
PONTUACAO = set('.,;:!?)]}"\'”’»…')
ABERTURA = set('([{"\'“‘«')


def normalizar(s):
    """Texto para comparação: sem acentos, sem espaços, minúsculo, com aspas e traços uniformes."""
    out = []
    for ch in unicodedata.normalize('NFKD', s):
        if unicodedata.combining(ch) or ch == '­':
            continue
        ch = EQUIV.get(ch, ch)
        if ch.isspace():
            continue
        out.append(ch.casefold())
    return ''.join(out)


def alnum(ch):
    return any(c.isalnum() for c in unicodedata.normalize('NFKD', ch))


# ---------------------------------------------------------------- tolerâncias

def tolerancias(manual=None):
    """(falta %, sobra %): parte de uma palavra que pode ficar sem destaque nas pontas e destaque
    admitido além da primeira ou da última letra, em % da altura da linha. Padrão 12 e 20."""
    return 12.0, 20.0


# ---------------------------------------------------------------- cor e máscaras

def hex_rgb(h):
    h = h.strip().lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def cor_dominante(arr):
    """Cor do marca-texto: a cor saturada e clara mais frequente da imagem (ou None)."""
    a = arr.astype(np.int16)
    mx, mn = a.max(axis=2), a.min(axis=2)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    cand = (lum > 150) & ((mx - mn) > 150)
    if cand.sum() < 150:
        return None
    q = (a[cand] // 16).astype(np.int32)
    chaves = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    valores, contagem = np.unique(chaves, return_counts=True)
    k = int(np.argmax(contagem))
    if contagem[k] < 150:
        return None
    sel = chaves == valores[k]
    return tuple(int(v) for v in np.median(a[cand][sel], axis=0))


def mascaras(arr, cor=None):
    """(destaque, fundo) de uma imagem RGB (numpy uint8).

    destaque: pixels da cor do marca-texto em trechos horizontais contínuos (descarta as franjas
    coloridas da suavização do texto); fundo: papel branco ou destaque (onde não há tinta).
    """
    a = arr.astype(np.int16)
    mx, mn = a.max(axis=2), a.min(axis=2)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    branco = (lum > 200) & ((mx - mn) < 60)
    cor = cor or cor_dominante(arr)
    if cor is None:
        return np.zeros(lum.shape, dtype=bool), branco
    dist = np.sqrt(((a.astype(np.int32) - np.array(cor, dtype=np.int32)) ** 2).sum(axis=2))
    bruto = dist < 50
    # trechos horizontais de ao menos 3 pixels
    d = bruto.copy()
    for k in (1, 2):
        d[:, :-k] &= bruto[:, k:]
    destaque = d.copy()
    for k in (1, 2):
        destaque[:, k:] |= d[:, :-k]
    return destaque, branco | destaque


def abrir_rgb(origem):
    im = origem if isinstance(origem, Image.Image) else Image.open(origem)
    im.load()
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        fundo = Image.new('RGBA', im.size, (255, 255, 255, 255))
        fundo.alpha_composite(im)
        im = fundo
    return im.convert('RGB')


# ---------------------------------------------------------------- OCR

def tesseract_disponivel():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class OcrSemPortugues(SystemExit):
    """O Tesseract não tem o idioma português: o destaque por OCR não é preciso o bastante."""


_IDIOMA_OCR = None


def _idiomas_tesseract():
    import pytesseract
    try:
        return set(pytesseract.get_languages(config=''))
    except Exception:
        return set()


def instalar_portugues():
    """Tenta instalar o idioma português do Tesseract (apt-get, sem interação). Devolve True se conseguiu."""
    import shutil
    import subprocess
    if not shutil.which('apt-get'):
        return False
    amb = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
    for cmds in ([['apt-get', 'install', '-y', '-qq', 'tesseract-ocr-por']],
                 [['apt-get', 'update', '-qq'], ['apt-get', 'install', '-y', '-qq', 'tesseract-ocr-por']]):
        try:
            for c in cmds:
                subprocess.run(c, check=True, timeout=240, env=amb,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        if 'por' in _idiomas_tesseract():
            return True
    return False


def idioma_ocr():
    """Sempre 'por'. Sem o idioma português, tenta instalá-lo; não conseguindo, para com erro (nunca usa inglês)."""
    global _IDIOMA_OCR
    if _IDIOMA_OCR:
        return _IDIOMA_OCR
    if 'por' not in _idiomas_tesseract():
        sys.stderr.write('OCR sem o idioma português: instalando tesseract-ocr-por...\n')
        if not instalar_portugues():
            raise OcrSemPortugues('ERRO: o Tesseract está sem o idioma português e a instalação automática falhou. '
                                  'Instalar ("apt-get install -y tesseract-ocr-por") antes de destacar ou conferir '
                                  'destaques; não usar OCR em inglês (módulo de colagens).')
    _IDIOMA_OCR = 'por'
    return _IDIOMA_OCR


class Palavra:
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'texto', 'linha', 'estado', 'cobertura', 'esq', 'dir', 'buraco',
                 'altura_coberta')

    def __init__(self, x0, y0, x1, y1, texto, linha):
        self.x0, self.y0, self.x1, self.y1, self.texto, self.linha = x0, y0, x1, y1, texto, linha
        self.estado = 'fora'
        self.cobertura = 0.0
        self.esq = self.dir = 0
        self.buraco = False
        self.altura_coberta = 0.0


class Linha:
    __slots__ = ('chave', 'x0', 'y0', 'x1', 'y1', 'palavras')

    def __init__(self, chave):
        self.chave = chave
        self.x0 = self.y0 = 10 ** 9
        self.x1 = self.y1 = -1
        self.palavras = []

    @property
    def altura(self):
        return max(1, self.y1 - self.y0)


def ocr_palavras(rgb):
    """Palavras e linhas (coordenadas na imagem original) pelo Tesseract."""
    import pytesseract
    w, h = rgb.size
    destaque, _ = mascaras(np.asarray(rgb))
    g = np.asarray(rgb.convert('L')).copy()
    g[destaque] = 255  # fundo destacado vira branco para o OCR
    esc = 3 if w < 1200 else (2 if w < 2400 else 1)
    gi = Image.fromarray(g)
    if esc > 1:
        gi = gi.resize((w * esc, h * esc), Image.LANCZOS)
    d = pytesseract.image_to_data(gi, lang=idioma_ocr(), config='--psm 6', output_type=pytesseract.Output.DICT)
    linhas = {}
    for i, t in enumerate(d['text']):
        nivel = d['level'][i]
        chave = (d['block_num'][i], d['par_num'][i], d['line_num'][i])
        x0, y0 = d['left'][i] / esc, d['top'][i] / esc
        x1, y1 = x0 + d['width'][i] / esc, y0 + d['height'][i] / esc
        if nivel == 4:
            ln = linhas.setdefault(chave, Linha(chave))
            ln.x0, ln.y0, ln.x1, ln.y1 = x0, y0, x1, y1
        elif nivel == 5 and t.strip():
            ln = linhas.setdefault(chave, Linha(chave))
            ln.palavras.append(Palavra(x0, y0, x1, y1, t.strip(), chave))
    saida = [ln for ln in linhas.values() if ln.palavras]
    for ln in saida:
        ln.palavras.sort(key=lambda p: p.x0)
        ln.x0 = min(ln.x0, min(p.x0 for p in ln.palavras))
        ln.x1 = max(ln.x1, max(p.x1 for p in ln.palavras))
        ln.y0 = min(ln.y0, min(p.y0 for p in ln.palavras))
        ln.y1 = max(ln.y1, max(p.y1 for p in ln.palavras))
    saida.sort(key=lambda ln: (ln.y0 + ln.y1) / 2)
    return saida


# ---------------------------------------------------------------- conferência nos pixels

def _preencher(defs, vals):
    """Colunas sem fundo suficiente herdam o estado da vizinha."""
    out = vals.copy()
    idx = np.nonzero(defs)[0]
    if idx.size == 0:
        return out
    ultimo = None
    for i in range(len(out)):
        if defs[i]:
            ultimo = out[i]
        elif ultimo is not None:
            out[i] = ultimo
    out[:idx[0]] = out[idx[0]]
    return out


def _colunas(destaque, fundo_img, x0, y0, x1, y1):
    """Por coluna da caixa: destacada (fundo majoritariamente colorido) ou não."""
    sub_d, fundo = destaque[y0:y1, x0:x1], fundo_img[y0:y1, x0:x1]
    if sub_d.size == 0:
        return np.zeros(0, dtype=bool)
    nfundo = fundo.sum(axis=0)
    ndest = (sub_d & fundo).sum(axis=0)
    defs = nfundo >= max(1, int(0.25 * (y1 - y0)))
    vals = np.where(nfundo > 0, ndest / np.maximum(nfundo, 1) > 0.5, False)
    return _preencher(defs, vals)


def _runs(vals):
    """Trechos contínuos True: [(início, fim exclusivo)]."""
    runs, ini = [], None
    for i, v in enumerate(vals):
        if v and ini is None:
            ini = i
        elif not v and ini is not None:
            runs.append((ini, i))
            ini = None
    if ini is not None:
        runs.append((ini, len(vals)))
    return runs


def conferir_destaques(origem, rotulo='imagem', tol_falta_pct=12, tol_sobra_pct=20, lupa=None):
    """Confere os destaques de uma imagem. Retorna (erros, avisos, infos, tem_destaque).

    Erros: palavra parcialmente destacada; destaque com sobra onde não há texto; destaque
    sobre área sem texto; destaque que invade palavra de outra linha.
    Avisos: destaque interrompido na quebra de linha; destaque que não cobre a altura das letras.
    """
    erros, avisos, infos = [], [], []
    rgb = abrir_rgb(origem)
    arr = np.asarray(rgb)
    destaque, fundo_img = mascaras(arr)
    if destaque.sum() < 30:
        return erros, avisos, infos, False
    if not tesseract_disponivel():
        avisos.append('%s: destaques não conferidos nos pixels (Tesseract indisponível); instalar com '
                      '"apt-get install -y tesseract-ocr tesseract-ocr-por" e "pip install pytesseract" '
                      'e, até lá, conferir na lupa letra a letra.' % rotulo)
        return erros, avisos, infos, True
    linhas = ocr_palavras(rgb)
    H, W = destaque.shape
    usados = np.zeros_like(destaque)
    faixas_lupa = []
    for n, ln in enumerate(linhas, 1):
        alt = ln.altura
        tol_falta = max(1, int(round(alt * tol_falta_pct / 100.0)))
        tol_sobra = max(2, int(round(alt * tol_sobra_pct / 100.0)))
        for p in ln.palavras:
            x0, x1 = int(np.floor(p.x0)), int(np.ceil(p.x1))
            y0, y1 = int(np.floor(p.y0)), int(np.ceil(p.y1))
            # altura inteira da palavra: acima e abaixo das letras o fundo revela o destaque
            cols = _colunas(destaque, fundo_img, max(0, x0), y0, min(W, x1), max(y0 + 1, y1))
            if cols.size == 0:
                continue
            ncol = int(cols.sum())
            p.cobertura = ncol / float(cols.size)
            if ncol <= tol_falta:
                # fora do destaque; confere se o destaque de outra linha invade a palavra
                sub_d, fundo = destaque[y0:y1, x0:x1], fundo_img[y0:y1, x0:x1]
                lin = (sub_d & fundo).sum(axis=1) / np.maximum(fundo.sum(axis=1), 1)
                frac = float((lin > 0.5).mean()) if lin.size else 0.0
                if frac >= 0.3:
                    erros.append('%s, linha %d: o destaque invade a palavra "%s", que não faz parte do trecho '
                                 '(%d%% da altura dela); a faixa de uma linha não pode cobrir a de cima ou a de baixo.'
                                 % (rotulo, n, p.texto, round(frac * 100)))
                p.estado = 'fora'
                continue
            idx = np.nonzero(cols)[0]
            p.esq, p.dir = int(idx[0]), int(cols.size - 1 - idx[-1])
            miolo = cols[idx[0]:idx[-1] + 1]
            p.buraco = any(b - a > max(2, tol_falta) for a, b in _runs(~miolo))
            folga_pont = int(round(alt * 0.4))
            esq_ok = p.esq <= tol_falta or (p.texto[:1] in ABERTURA and p.esq <= folga_pont)
            dir_ok = p.dir <= tol_falta or (p.texto[-1:] in PONTUACAO and p.dir <= folga_pont)
            lin_d = destaque[y0:y1, x0:x1] & fundo_img[y0:y1, x0:x1]
            fr = lin_d.sum(axis=1) / np.maximum(fundo_img[y0:y1, x0:x1].sum(axis=1), 1)
            linhas_d = fr > 0.5
            p.altura_coberta = float(linhas_d.mean()) if fr.size else 0.0
            terco = max(1, linhas_d.size // 3)
            topo, base = float(linhas_d[:terco].mean()), float(linhas_d[-terco:].mean())
            if (topo < 0.25 <= 0.75 <= base) or (base < 0.25 <= 0.75 <= topo):
                erros.append('%s, linha %d: o destaque cobre só a parte de %s de "%s"; a faixa está desalinhada ou '
                             'vem da linha vizinha e deve cobrir a altura inteira da linha, sem invadir as outras.'
                             % (rotulo, n, 'baixo' if base > topo else 'cima', p.texto))
            if esq_ok and dir_ok and not p.buraco:
                p.estado = 'dentro'
            else:
                p.estado = 'parcial'
                partes = []
                if not esq_ok:
                    partes.append('o início ficou sem destaque (%d px)' % p.esq)
                if not dir_ok:
                    partes.append('o final ficou sem destaque (%d px)' % p.dir)
                if p.buraco:
                    partes.append('há falha no meio do destaque')
                erros.append('%s, linha %d: palavra parcialmente destacada "%s" — %s; o destaque deve cobrir a '
                             'palavra inteira, da primeira à última letra.' % (rotulo, n, p.texto, '; '.join(partes)))
        # sobras e destaques sem texto na faixa da linha
        y0, y1 = int(np.floor(ln.y0)), int(np.ceil(ln.y1))
        h = max(1, y1 - y0)
        cy0, cy1 = y0 + int(0.25 * h), max(y0 + int(0.25 * h) + 1, y1 - int(0.25 * h))
        cols = _colunas(destaque, fundo_img, 0, cy0, W, cy1)
        for a, b in _runs(cols):
            if b - a <= tol_falta:
                continue
            usados[max(0, y0 - int(0.6 * alt)):y1 + int(0.6 * alt), a:b] = True
            dentro = [p for p in ln.palavras if p.estado in ('dentro', 'parcial') and p.x1 > a and p.x0 < b]
            if not dentro:
                if b - a > 0.5 * alt:
                    erros.append('%s, linha %d: destaque sobre área sem texto (%d px de largura); remover.'
                                 % (rotulo, n, b - a))
                continue
            sobra_esq = dentro[0].x0 - a
            sobra_dir = b - dentro[-1].x1
            if sobra_esq > tol_sobra:
                erros.append('%s, linha %d: o destaque começa %d px antes de "%s", onde não há texto; '
                             'deve começar na primeira letra.' % (rotulo, n, round(sobra_esq), dentro[0].texto))
            if sobra_dir > tol_sobra:
                erros.append('%s, linha %d: o destaque passa %d px de "%s", onde não há texto; '
                             'deve terminar na última letra.' % (rotulo, n, round(sobra_dir), dentro[-1].texto))
            faixas_lupa.append((n, a, b, y0, y1))
        baixos = [p for p in ln.palavras if p.estado == 'dentro' and p.altura_coberta < 0.6]
        if baixos:
            avisos.append('%s, linha %d: o destaque não cobre a altura das letras em %s; a faixa deve cobrir '
                          'a linha inteira, sem invadir as vizinhas.'
                          % (rotulo, n, ', '.join('"%s"' % p.texto for p in baixos[:4])))
    # interrupções na quebra de linha (trecho contínuo com palavras de fora)
    for i in range(len(linhas) - 1):
        a, b = linhas[i], linhas[i + 1]
        if (b.y0 - a.y1) > 1.2 * max(a.altura, b.altura):
            continue
        ea = [p.estado != 'fora' for p in a.palavras]
        eb = [p.estado != 'fora' for p in b.palavras]
        if any(ea) and not ea[-1] and eb and eb[0]:
            ult = max(k for k, v in enumerate(ea) if v)
            fora = [p.texto for p in a.palavras[ult + 1:]]
            avisos.append('%s, linhas %d-%d: o destaque é interrompido antes do fim da linha e retomado no início '
                          'da seguinte; ficaram sem destaque: "%s". Conferir se o trecho é contínuo.'
                          % (rotulo, i + 1, i + 2, ' '.join(fora)))
        if ea and ea[-1] and any(eb) and not eb[0]:
            pri = min(k for k, v in enumerate(eb) if v)
            fora = [p.texto for p in b.palavras[:pri]]
            avisos.append('%s, linhas %d-%d: o destaque vai até o fim da linha e só é retomado depois do início '
                          'da seguinte; ficaram sem destaque: "%s". Conferir se o trecho é contínuo.'
                          % (rotulo, i + 1, i + 2, ' '.join(fora)))
    # destaques que não pertencem a nenhuma linha de texto (manchas soltas)
    rotulos = _componentes(destaque)
    if rotulos is None:
        infos.append('%s: manchas de destaque sem texto não conferidas (instalar opencv-python ou scipy).' % rotulo)
    else:
        total = np.bincount(rotulos.ravel())
        dentro_faixa = np.bincount(rotulos[usados & destaque].ravel(), minlength=total.size)
        manchas = []
        for k in range(1, total.size):
            if total[k] < 40 or dentro_faixa[k] >= 0.5 * total[k]:
                continue
            ys, xs = np.nonzero(rotulos == k)
            if (xs.max() - xs.min()) >= 6:
                manchas.append('x %d-%d, y %d-%d' % (xs.min(), xs.max(), ys.min(), ys.max()))
        if manchas:
            erros.append('%s: %d mancha(s) de destaque sem texto (%s%s); remover.'
                         % (rotulo, len(manchas), '; '.join(manchas[:3]), '; ...' if len(manchas) > 3 else ''))
    infos.append('%s: %d linha(s) de texto, %d palavra(s) destacada(s) conferida(s) por OCR.'
                 % (rotulo, len(linhas), sum(1 for ln in linhas for p in ln.palavras if p.estado != 'fora')))
    if lupa:
        gerar_lupa(rgb, faixas_lupa, lupa)
        infos.append('Lupa gerada em: %s' % lupa)
    return erros, avisos, infos, True


def _componentes(mascara):
    """Rótulos dos componentes conexos da máscara (opencv ou scipy)."""
    try:
        import cv2
        _, rot = cv2.connectedComponents(mascara.astype(np.uint8), connectivity=8)
        return rot
    except ImportError:
        pass
    try:
        from scipy import ndimage
        rot, _ = ndimage.label(mascara, structure=np.ones((3, 3)))
        return rot
    except ImportError:
        return None


def gerar_lupa(rgb, faixas, caminho, alvo_px=90):
    """Pontas (início e fim) de cada faixa destacada, ampliadas lado a lado, uma linha por faixa."""
    if not faixas:
        return None
    W, H = rgb.size
    tiras = []
    for (n, a, b, y0, y1) in faixas:
        alt = max(1, y1 - y0)
        zoom = max(1, min(6, int(round(alvo_px / float(alt)))))
        m = int(alt * 0.3)
        yy0, yy1 = max(0, y0 - m), min(H, y1 + m)
        fora, dentro = int(alt * 2.5), int(alt * 5)
        e = rgb.crop((max(0, a - fora), yy0, min(W, a + dentro), yy1))
        d = rgb.crop((max(0, b - dentro), yy0, min(W, b + fora), yy1))
        e = e.resize((e.width * zoom, e.height * zoom), Image.NEAREST)
        d = d.resize((d.width * zoom, d.height * zoom), Image.NEAREST)
        tira = Image.new('RGB', (e.width + d.width + 16, max(e.height, d.height)), (110, 110, 110))
        tira.paste(e, (0, 0))
        tira.paste(d, (e.width + 16, 0))
        tiras.append(tira)
    largura = max(t.width for t in tiras)
    altura = sum(t.height for t in tiras) + 6 * (len(tiras) - 1)
    folha = Image.new('RGB', (largura, altura), (110, 110, 110))
    y = 0
    for t in tiras:
        folha.paste(t, (0, y))
        y += t.height + 6
    folha.save(caminho)
    return caminho


# ---------------------------------------------------------------- PDF com camada de texto

class Car:
    __slots__ = ('t', 'x0', 'x1', 'top', 'bottom', 'size', 'linha', 'palavra', 'eol_hifen')

    def __init__(self, c):
        self.t = c['text']
        self.x0, self.x1, self.top, self.bottom = float(c['x0']), float(c['x1']), float(c['top']), float(c['bottom'])
        self.size = float(c.get('size') or (self.bottom - self.top) or 10)
        self.linha = self.palavra = -1
        self.eol_hifen = False


def linhas_do_pdf(page):
    cars = [Car(c) for c in page.chars if c.get('upright', True) and c['text'] and not c['text'].isspace()]
    cars.sort(key=lambda c: (c.bottom, c.x0))
    linhas = []
    for c in cars:
        alvo = None
        for ln in reversed(linhas[-3:]):
            base = sorted(x.bottom for x in ln)[len(ln) // 2]
            if abs(c.bottom - base) <= 0.5 * c.size:
                alvo = ln
                break
        if alvo is None:
            linhas.append([c])
        else:
            alvo.append(c)
    linhas.sort(key=lambda ln: sorted(x.bottom for x in ln)[len(ln) // 2])
    resultado = []
    for i, ln in enumerate(linhas):
        ln.sort(key=lambda c: c.x0)
        # negrito simulado (mesma letra impressa duas vezes, levemente deslocada): manter uma só
        unicos = []
        for c in ln:
            if unicos and c.t == unicos[-1].t and abs(c.x0 - unicos[-1].x0) < 0.15 * c.size:
                unicos[-1].x1 = max(unicos[-1].x1, c.x1)
                continue
            unicos.append(c)
        ln[:] = unicos
        palavras, atual = [], []
        for c in ln:
            if atual and (c.x0 - atual[-1].x1) > max(0.12 * c.size, 0.8):
                palavras.append(atual)
                atual = []
            atual.append(c)
        if atual:
            palavras.append(atual)
        for j, pw in enumerate(palavras):
            for c in pw:
                c.linha, c.palavra = i, j
        resultado.append(palavras)
    # hífen de translineação: último caractere da linha, depois de letra, com a linha seguinte em minúscula
    for i in range(len(resultado) - 1):
        ult = resultado[i][-1][-1]
        prox = resultado[i + 1][0][0]
        if ult.t in HIFENS and len(resultado[i][-1]) > 1 and alnum(resultado[i][-1][-2].t) and prox.t[:1].islower():
            ult.eol_hifen = True
    return resultado


def fluxo(linhas, sem_hifen_eol):
    """Texto normalizado contínuo e o mapa índice normalizado -> caractere de origem."""
    s, mapa = [], []
    for palavras in linhas:
        for pw in palavras:
            for c in pw:
                if sem_hifen_eol and c.eol_hifen:
                    continue
                for ch in normalizar(c.t):
                    s.append(ch)
                    mapa.append(c)
    return ''.join(s), mapa


def localizar(linhas, trecho, ocorrencia=None):
    alvo = normalizar(trecho)
    if not alvo:
        raise ValueError('trecho vazio')
    for sem_hifen in (False, True):
        s, mapa = fluxo(linhas, sem_hifen)
        achados, i = [], s.find(alvo)
        while i >= 0:
            achados.append(i)
            i = s.find(alvo, i + 1)
        if achados:
            break
    if not achados:
        s, _ = fluxo(linhas, True)
        sm = difflib.SequenceMatcher(None, s, alvo, autojunk=False)
        bloco = max(sm.get_matching_blocks(), key=lambda b: b.size)
        ini = max(0, bloco.a - bloco.b)
        sugestao = s[ini:ini + len(alvo) + 10]
        raise LookupError('trecho não localizado na página. Trecho mais parecido (normalizado, sem espaços): "%s". '
                          'Copiar o trecho literalmente do documento; não aproximar.' % sugestao)
    if len(achados) > 1 and not ocorrencia:
        raise LookupError('o trecho aparece %d vezes na página; indicar --ocorrencia (1 a %d) ou ampliar o trecho.'
                          % (len(achados), len(achados)))
    k = (ocorrencia or 1) - 1
    if k >= len(achados):
        raise LookupError('ocorrência %d inexistente (há %d).' % (k + 1, len(achados)))
    i = achados[k]
    cars = []
    for c in mapa[i:i + len(alvo)]:
        if not cars or cars[-1] is not c:
            cars.append(c)
    return cars


def expandir_palavras(linhas, cars, avisos):
    """Nunca destacar parte de palavra: estende o início e o fim às letras inteiras da palavra."""
    sel = list(cars)
    pri, ult = sel[0], sel[-1]
    pw = linhas[pri.linha][pri.palavra]
    k = pw.index(pri)
    letras = [j for j in range(k) if alnum(pw[j].t)]
    if letras and alnum(pri.t):
        sel = pw[letras[0]:k] + sel
        avisos.append('o trecho começava no meio de "%s"; o destaque foi estendido à palavra inteira.'
                      % ''.join(c.t for c in pw))
    pw = linhas[ult.linha][ult.palavra]
    k = pw.index(ult)
    letras = [j for j in range(k + 1, len(pw)) if alnum(pw[j].t)]
    if letras and alnum(ult.t):
        sel = sel + pw[k + 1:letras[-1] + 1]
        avisos.append('o trecho terminava no meio de "%s"; o destaque foi estendido à palavra inteira.'
                      % ''.join(c.t for c in pw))
    return sel


def retangulos_pdf(linhas, sel):
    """Um retângulo por linha: da primeira à última letra destacada, na altura da linha."""
    por_linha = {}
    for c in sel:
        por_linha.setdefault(c.linha, []).append(c)
    nums = sorted(por_linha)
    rets = []
    for n in nums:
        cs = por_linha[n]
        x0, x1 = min(c.x0 for c in cs), max(c.x1 for c in cs)
        ult = linhas[n][-1][-1]
        if ult.eol_hifen and (n + 1) in por_linha and ult not in cs:
            x1 = max(x1, ult.x1)
        todos = [c for pw in linhas[n] for c in pw]
        top, bottom = min(c.top for c in todos), max(c.bottom for c in todos)
        size = max(c.size for c in todos)
        rets.append([n, x0, top, x1, bottom, size])
    # sem invadir a linha vizinha: limita ao meio do intervalo entre linhas
    tops = {i: min(c.top for pw in ln for c in pw) for i, ln in enumerate(linhas)}
    bots = {i: max(c.bottom for pw in ln for c in pw) for i, ln in enumerate(linhas)}
    for r in rets:
        n, size = r[0], r[5]
        folga = 0.04 * size
        respiro = 0.06 * size  # separação visível entre as faixas de linhas consecutivas
        cima = r[2] - folga
        if n - 1 in bots and bots[n - 1] < tops[n]:
            cima = max(cima, (bots[n - 1] + tops[n]) / 2 + respiro / 2)
        baixo = r[4] + folga
        if n + 1 in tops and tops[n + 1] > bots[n]:
            baixo = min(baixo, (bots[n] + tops[n + 1]) / 2 - respiro / 2)
        r[2], r[4] = cima, baixo
    return rets


def conferir_geometria(linhas, sel, rets, erros):
    """Nenhum caractere fora do trecho dentro de um retângulo; todo caractere do trecho coberto."""
    dentro = set(id(c) for c in sel)
    for n, x0, top, x1, bottom, size in rets:
        for pw in linhas[n]:
            for c in pw:
                sobre = min(x1, c.x1) - max(x0, c.x0)
                if id(c) in dentro:
                    if c.x0 < x0 - 0.01 or c.x1 > x1 + 0.01:
                        erros.append('geometria: a letra "%s" do trecho não está inteiramente coberta.' % c.t)
                elif sobre > 0.25 and not c.eol_hifen:
                    erros.append('geometria: o destaque cobre "%s", que não faz parte do trecho.' % c.t)


def destacar_pdf(a):
    import pdfplumber
    erros, avisos = [], []
    with pdfplumber.open(a.pdf) as pdf:
        if not 1 <= a.pagina <= len(pdf.pages):
            raise SystemExit('Página %d inexistente (o PDF tem %d).' % (a.pagina, len(pdf.pages)))
        page = pdf.pages[a.pagina - 1]
        linhas = linhas_do_pdf(page)
        im = page.to_image(resolution=a.dpi)
        base = im.original.convert('RGB')
        escala = im.scale
        ox, oy = float(page.bbox[0]), float(page.bbox[1])
        if not linhas or a.ocr:
            if not linhas:
                avisos.append('a página não tem camada de texto (documento digitalizado); destaque por OCR.')
            return destacar_imagem_rgb(base, a, erros, avisos)
        area = float(page.width * page.height) or 1.0
        if any((float(i['x1']) - float(i['x0'])) * (float(i['bottom']) - float(i['top'])) > 0.6 * area
               for i in page.images):
            avisos.append('página digitalizada com camada de texto de OCR: as caixas das letras podem não coincidir com a '
                          'imagem; se a conferência acusar erro, repetir com --ocr.')
        todos_sel, todos_rets = [], []
        for t in a.trecho:
            cars = localizar(linhas, t, a.ocorrencia)
            sel = expandir_palavras(linhas, cars, avisos)
            rets = retangulos_pdf(linhas, sel)
            conferir_geometria(linhas, sel, rets, erros)
            texto = _texto_sel(linhas, sel)
            print('Trecho localizado (%d linha(s)): %s' % (len(rets), texto))
            todos_sel.extend(sel)
            todos_rets.extend(rets)
    # pintura (multiplicação: o texto continua preto e legível)
    arr = np.asarray(base).astype(np.float32)
    cor = np.array(hex_rgb(a.cor), dtype=np.float32) / 255.0
    for n, x0, top, x1, bottom, size in todos_rets:
        px0, px1 = int(np.floor((x0 - ox) * escala)), int(np.ceil((x1 - ox) * escala))
        py0, py1 = int(np.floor((top - oy) * escala)), int(np.ceil((bottom - oy) * escala))
        arr[py0:py1, px0:px1, :] *= cor
    pintada = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    # recorte
    if a.recorte == 'pagina':
        caixa = (0, 0, pintada.width, pintada.height)
    elif a.recorte:
        x0, t0, x1, b0 = [float(v) for v in a.recorte.split(',')]
        caixa = (int((x0 - ox) * escala), int((t0 - oy) * escala), int((x1 - ox) * escala), int((b0 - oy) * escala))
    else:
        nums = sorted(set(r[0] for r in todos_rets))
        li, lf = max(0, nums[0] - a.linhas_antes), min(len(linhas) - 1, nums[-1] + a.linhas_depois)
        cs = [c for ln in linhas[li:lf + 1] for pw in ln for c in pw]
        size = max(c.size for c in cs)
        m = 0.3 * size
        caixa = (int((min(c.x0 for c in cs) - m - ox) * escala), int((min(c.top for c in cs) - m - oy) * escala),
                 int(np.ceil((max(c.x1 for c in cs) + m - ox) * escala)),
                 int(np.ceil((max(c.bottom for c in cs) + m - oy) * escala)))
    caixa = (max(0, caixa[0]), max(0, caixa[1]), min(pintada.width, caixa[2]), min(pintada.height, caixa[3]))
    recorte = pintada.crop(caixa)
    recorte.save(a.saida, dpi=(a.dpi, a.dpi))
    return finalizar(a, recorte, erros, avisos)


def _texto_sel(linhas, sel):
    partes = []
    for grp in _agrupar_palavras(sel):
        t = ''.join(c.t for c in grp)
        ult = linhas[grp[-1].linha][-1][-1]
        if ult.eol_hifen and grp[-1] is linhas[grp[-1].linha][-1][-2] and ult not in grp:
            t += '-'
        partes.append(t)
    return ' '.join(partes).replace('- ', '-')


def _agrupar_palavras(sel):
    grupos, atual, chave = [], [], None
    for c in sel:
        k = (c.linha, c.palavra)
        if chave is not None and k != chave and not (atual and atual[-1].eol_hifen):
            grupos.append(atual)
            atual = []
        atual.append(c)
        chave = k
    if atual:
        grupos.append(atual)
    return grupos


# ---------------------------------------------------------------- imagem (OCR)

def destacar_imagem_rgb(rgb, a, erros, avisos):
    if not tesseract_disponivel():
        raise SystemExit('Tesseract indisponível: instalar ("apt-get install -y tesseract-ocr tesseract-ocr-por" e '
                         '"pip install pytesseract") ou obter o PDF original com camada de texto. '
                         'Não desenhar destaque por estimativa.')
    linhas = ocr_palavras(rgb)
    seq = [(i, p) for i, ln in enumerate(linhas) for p in ln.palavras]
    toks = [normalizar(re.sub(r'[^\w]', '', p.texto)) for _, p in seq]
    rets = []
    for t in a.trecho:
        alvo = [normalizar(re.sub(r'[^\w]', '', w)) for w in t.split()]
        alvo = [w for w in alvo if w]
        alvo_s = ''.join(alvo)
        melhores = []
        n = len(alvo)
        inicios = [i for i in range(len(seq))
                   if any(difflib.SequenceMatcher(None, toks[j], alvo[0]).ratio() >= 0.6
                          for j in range(i, min(i + 2, len(seq))))] if alvo else []
        for ini in inicios:
            for tam in range(max(1, n - 3), n + 4):
                if ini + tam > len(seq):
                    break
                cand = ''.join(toks[ini:ini + tam])
                r = difflib.SequenceMatcher(None, cand, alvo_s, autojunk=False).ratio()
                # pontas: a primeira e a última palavra lidas devem corresponder às do trecho
                pontas = min(difflib.SequenceMatcher(None, toks[ini], alvo[0]).ratio(),
                             difflib.SequenceMatcher(None, toks[ini + tam - 1], alvo[-1]).ratio())
                if pontas < 0.5:
                    continue
                melhores.append((r - 0.02 * abs(tam - n), ini, tam))
        melhores.sort(reverse=True)
        if not melhores or melhores[0][0] < 0.85:
            raise LookupError('trecho não localizado com segurança no OCR (semelhança máxima %.2f). Conferir o '
                              'trecho ou obter o PDF original com camada de texto.' % (melhores[0][0] if melhores else 0))
        piso = max(0.85, melhores[0][0] - 0.03)
        achados = []
        for r, ini, tam in melhores:
            if r < piso:
                break
            if all(ini + tam <= i2 or ini >= i2 + t2 for _, i2, t2 in achados):
                achados.append((r, ini, tam))
        achados.sort(key=lambda m: m[1])
        if len(achados) > 1 and not a.ocorrencia:
            raise LookupError('o trecho aparece %d vezes na imagem; indicar --ocorrencia (1 a %d) ou ampliar o trecho.'
                              % (len(achados), len(achados)))
        k = (a.ocorrencia or 1) - 1
        if k >= len(achados):
            raise LookupError('ocorrência %d inexistente (há %d).' % (k + 1, len(achados)))
        r, ini, tam = achados[k]
        escolhidas = seq[ini:ini + tam]
        lido = ' '.join(p.texto for _, p in escolhidas)
        print('Trecho localizado por OCR (semelhança %.2f): %s' % (r, lido))
        avisos.append('conferir na lupa, letra a letra, se o OCR leu "%s" como no documento.' % lido)
        palavras_t = t.split()
        ini_x = _aparar(np.asarray(rgb), escolhidas[0][1], palavras_t[0], inicio=True, avisos=avisos)
        fim_x = _aparar(np.asarray(rgb), escolhidas[-1][1], palavras_t[-1], inicio=False, avisos=avisos)
        por_linha = {}
        for i, p in escolhidas:
            por_linha.setdefault(i, []).append(p)
        for i, ps in sorted(por_linha.items()):
            ln = linhas[i]
            x0 = min(p.x0 for p in ps)
            x1 = max(p.x1 for p in ps)
            if escolhidas[0][1] in ps:
                x0 = ini_x
            if escolhidas[-1][1] in ps:
                x1 = fim_x
            rets.append((i, x0, ln.y0, x1, ln.y1, ln.altura))
    arr = np.asarray(rgb).astype(np.float32)
    cor = np.array(hex_rgb(a.cor), dtype=np.float32) / 255.0
    H, W = arr.shape[:2]
    for i, x0, y0, x1, y1, alt in rets:
        folga_h, folga_v = 0.08 * alt, 0.12 * alt
        cima = (linhas[i - 1].y1 + y0) / 2 if i > 0 and linhas[i - 1].y1 < y0 else y0
        baixo = (y1 + linhas[i + 1].y0) / 2 if i + 1 < len(linhas) and linhas[i + 1].y0 > y1 else y1
        py0, py1 = int(max(cima, y0 - folga_v)), int(np.ceil(min(baixo, y1 + folga_v)))
        px0, px1 = int(max(0, x0 - folga_h)), int(min(W, np.ceil(x1 + folga_h)))
        arr[py0:py1, px0:px1, :] *= cor
    pintada = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    pintada.save(a.saida)
    return finalizar(a, pintada, erros, avisos)


def _aparar(arr, p, palavra_trecho, inicio, avisos):
    """Exclui do destaque a pontuação colada à palavra no OCR que não faz parte do trecho."""
    lido = p.texto
    if inicio:
        sobra = len(lido) - len(lido.lstrip(''.join(ABERTURA)))
        sobra -= len(palavra_trecho) - len(palavra_trecho.lstrip(''.join(ABERTURA)))
    else:
        sobra = len(lido) - len(lido.rstrip(''.join(PONTUACAO)))
        sobra -= len(palavra_trecho) - len(palavra_trecho.rstrip(''.join(PONTUACAO)))
    borda = p.x0 if inicio else p.x1
    if sobra <= 0:
        return borda
    x0, x1 = int(np.floor(p.x0)), int(np.ceil(p.x1))
    y0, y1 = int(np.floor(p.y0)), int(np.ceil(p.y1))
    _, fundo = mascaras(arr[y0:y1, x0:x1])
    tinta = (~fundo).any(axis=0)
    runs = _runs(tinta)
    if len(runs) <= sobra:
        avisos.append('não foi possível separar a pontuação de "%s"; conferir na lupa.' % lido)
        return borda
    if inicio:
        return float(x0 + runs[sobra][0])
    return float(x0 + runs[-sobra - 1][1])


def finalizar(a, imagem, erros, avisos):
    lupa = a.lupa or os.path.splitext(a.saida)[0] + ' - lupa.png'
    falta, sobra = tolerancias()
    e2, v2, i2, _ = conferir_destaques(imagem, os.path.basename(a.saida), falta, sobra, lupa=lupa)
    erros += e2
    avisos += v2
    print('\nRecorte gravado em: %s' % a.saida)
    print('\nERROS (%d):' % len(erros))
    for e in erros or ['nenhum']:
        print('  - ' + e)
    print('\nAVISOS (%d):' % len(avisos))
    for v in avisos or ['nenhum']:
        print('  - ' + v)
    for i in i2:
        print('\nObservação: ' + i)
    print('\nConferência visual obrigatória: abrir a lupa e conferir o início e o fim de cada linha destacada.')
    if erros and a.pdf and not a.ocr:
        print('Sugestão: se a página for digitalizada, a camada de texto pode não coincidir com a imagem; '
              'repetir com --ocr e conferir de novo.')
    print('RESULTADO: %s' % ('APROVADO (sem erros)' if not erros else 'REPROVADO (%d erro(s)) — não usar este recorte'
                             % len(erros)))
    return 1 if erros else 0


def main():
    ap = argparse.ArgumentParser(description='Pixel-accurate highlights on excerpts (destaques precisos nos recortes).')
    ap.add_argument('--pdf', help='PDF de origem (com camada de texto ou digitalizado)')
    ap.add_argument('--pagina', type=int, default=1, help='página do PDF (1 = primeira)')
    ap.add_argument('--imagem', help='imagem de origem (print, foto, digitalização)')
    ap.add_argument('--trecho', action='append', default=[], help='texto literal a destacar (pode repetir)')
    ap.add_argument('--ocorrencia', type=int, help='qual ocorrência do trecho, se houver mais de uma')
    ap.add_argument('--saida', help='PNG de saída com o destaque')
    ap.add_argument('--recorte', help='"x0,topo,x1,base" em pontos do PDF, ou "pagina"; padrão: linhas do trecho')
    ap.add_argument('--linhas-antes', type=int, default=1, dest='linhas_antes')
    ap.add_argument('--linhas-depois', type=int, default=1, dest='linhas_depois')
    ap.add_argument('--dpi', type=int, default=220)
    ap.add_argument('--cor', default='FFFF00', help='cor do marca-texto (hex), padrão amarelo FFFF00')
    ap.add_argument('--ocr', action='store_true', help='no PDF, usar OCR mesmo havendo camada de texto')
    ap.add_argument('--conferir', metavar='IMAGEM', help='só conferir os destaques de uma imagem pronta')
    ap.add_argument('--lupa', help='arquivo da lupa (padrão: "<saída> - lupa.png")')
    a = ap.parse_args()

    if a.conferir:
        lupa = a.lupa or os.path.splitext(a.conferir)[0] + ' - lupa.png'
        falta, sobra = tolerancias()
        erros, avisos, infos, tem = conferir_destaques(a.conferir, os.path.basename(a.conferir), falta, sobra,
                                                       lupa=lupa)
        if not tem:
            print('Nenhum destaque encontrado na imagem.')
            return 0
        print('CONFERÊNCIA DE DESTAQUES — %s' % os.path.basename(a.conferir))
        print('\nERROS (%d):' % len(erros))
        for e in erros or ['nenhum']:
            print('  - ' + e)
        print('\nAVISOS (%d):' % len(avisos))
        for v in avisos or ['nenhum']:
            print('  - ' + v)
        for i in infos:
            print('\nObservação: ' + i)
        print('\nRESULTADO: %s' % ('APROVADO (sem erros)' if not erros else 'REPROVADO (%d erro(s))' % len(erros)))
        return 1 if erros else 0
    if not a.trecho or not a.saida or not (a.pdf or a.imagem):
        ap.error('informar --pdf ou --imagem, ao menos um --trecho e --saida (ou usar --conferir)')
    try:
        if a.pdf:
            return destacar_pdf(a)
        return destacar_imagem_rgb(abrir_rgb(a.imagem), a, [], [])
    except LookupError as exc:
        print('TRECHO: %s' % exc)
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print('FALHA: %s: %s' % (type(exc).__name__, exc))
        sys.exit(2)
