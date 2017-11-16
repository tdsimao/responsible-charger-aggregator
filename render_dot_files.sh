#!/usr/bin/env bash
for f in $(ls out/*.dot grids/*.dot); do echo render $f; dot $f -Tpdf -o $f.pdf; pdfcrop $f.pdf $f.pdf; done
