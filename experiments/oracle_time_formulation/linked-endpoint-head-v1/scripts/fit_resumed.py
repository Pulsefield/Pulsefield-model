"""Fit the unchanged paired protocol against the verified resumed manifest."""
import fit
from extract import OWNER

if __name__=="__main__":
    fit.FEATURES=OWNER/"features-rest"
    fit.main()
