from pdflatex import PDFLaTeX
import amazon.amazon_reviews
import breast.breast_cancer


pdfl = PDFLaTeX.from_texfile("main.tex")
pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)