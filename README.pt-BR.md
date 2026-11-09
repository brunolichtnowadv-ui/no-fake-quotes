# No Fake Quotes (sem citação falsa)

**Duas ferramentas pequenas que conferem o texto escrito com ajuda de IA contra as fontes antes de qualquer outra pessoa: um conferidor que pega citação inventada ou alterada e um marca-texto que destaca recortes com precisão de pixel.**

[English](README.md)

---

## Por quê

O erro mais grave na escrita profissional com IA é a citação que parece certa e não é: um julgado que não existe, uma lei com uma palavra trocada, um trecho parafraseado que manteve as aspas. Reler não pega com segurança. Comparar com a fonte, palavra por palavra, pega.

As ferramentas nasceram do uso diário num fluxo de serviços profissionais em que toda citação precisa bater com a fonte.

## 1. `quote_checker.py` (conferidor de citações)

```bash
python3 quote_checker.py rascunho.docx fontes/
python3 quote_checker.py rascunho.md lei.pdf acordao.pdf --ocr --lang por
```

- **O que é citação:** parágrafo em bloco recuado no `.docx`, citação `>` no `.md` ou trecho entre aspas com 8 palavras ou mais.
- **Como compara:** tira a referência final entre parênteses e marcas como `[grifou-se]`, divide nos cortes (`[...]`, `(...)`, `...`) e procura cada fragmento nas fontes, sem considerar acentos, maiúsculas e pontuação.
- **Resultado:** `OK` (achou, com arquivo e página), `DIVERGE` (há trecho parecido, com as palavras diferentes, para corrigir pela fonte) ou `NOT FOUND` (nenhuma fonte salva contém o trecho: achar e salvar a fonte, ou excluir a citação, nunca reconstruir de memória).
- O código de saída é `1` se algo não estiver OK, para travar a entrega numa skill ou num script.

## 2. `highlight.py` (marca-texto)

```bash
python3 highlight.py --pdf "Documento.pdf" --pagina 3 --trecho "trecho literal" --saida recorte.png
python3 highlight.py --imagem print.png --trecho "trecho literal" --saida recorte.png
```

Destaca pela posição real de cada letra e depois confere nos pixels, por OCR independente, que nenhuma palavra ficou pela metade, que não sobrou destaque em espaço vazio e que nada invadiu a linha de cima ou de baixo. Gera também uma "lupa" com as pontas de cada linha ampliadas.

## Limitações

- O conferidor confere a literalidade, não a pertinência, a vigência nem a identificação correta da fonte.
- PDF digitalizado exige `--ocr`, e erros de OCR podem gerar `DIVERGE` falso.
- Trechos entre aspas com menos de 8 palavras não são conferidos.

## Licença

[MIT](LICENSE)
