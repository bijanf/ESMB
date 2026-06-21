#!/bin/bash
# Build the Chronos-ESM manual to main.pdf
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
