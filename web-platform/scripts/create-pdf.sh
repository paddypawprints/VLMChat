#!/bin/bash
# PDF Creation Script for Whitepaper
# Requires pandoc to be installed

if ! command -v pandoc &> /dev/null; then
    echo "❌ Pandoc is not installed. Please install it first:"
    echo "📥 macOS: brew install pandoc"
    echo "📥 Ubuntu: sudo apt-get install pandoc"
    echo "📥 Windows: Download from https://pandoc.org/installing.html"
    exit 1
fi

echo "📄 Creating PDF whitepaper..."

pandoc whitepaper.md \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    -V toccolor=gray \
    --toc \
    --toc-depth=2 \
    -o whitepaper.pdf

if [ $? -eq 0 ]; then
    echo "✅ PDF created successfully: whitepaper.pdf"
else
    echo "❌ PDF creation failed. Check pandoc installation and dependencies."
fi