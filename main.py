from pdflatex import PDFLaTeX
import wine-quality.wine_quality


pdfl = PDFLaTeX.from_texfile("main.tex")
pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)