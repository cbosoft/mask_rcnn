import numpy as np


def gaussian_pdf(x, mu, std):
    pdf = 1./std/np.sqrt(2.*np.pi)*np.exp(-0.5*np.square((x - mu)/std))
    # not always normalised properly? binning issue?
    pdf /= np.trapz(pdf, x)
    return pdf
