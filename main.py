from pdflatex import PDFLaTeX
import amazon.amazon_reviews
import breast.breast_cancer
import iris.iris
import wine.wine_quality


pdfl = PDFLaTeX.from_texfile("main.tex")
pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)