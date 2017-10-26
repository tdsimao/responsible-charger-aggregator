for f in $(ls out/*.dot); do echo render $f; dot $f -Tpdf -o $f.pdf; done
