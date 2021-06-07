# Diagnostic scripts for evaluating land-atmosphere interactions in climate models. 

Author: Jess Baker (j.c.baker@leeds.ac.uk) 

The scripts in this repository were developed as part of the CSSP Brazil LAPSE project at the University of Leeds, UK, and accompany the following manuscript:

Baker, J. C. A., Castilho De Souza, D., Kubota, P. Y., Buermann, W., Coelho, C. a. S., Andrews, M. B., Gloor, M., Garcia-Carreras, L., Figueroa, S. N. & Spracklen, D. V. 2021. An Assessment of Land–Atmosphere Interactions over South America Using Satellites, Reanalysis, and Two Global Climate Models. Journal of Hydrometeorology, 22, 905-922.

Users of the land-atmosphere diagnostics are asked to please cite this publication and any diagnostic-specific references within.


Diagnostic scripts:
- terrestrial_coupling_index.py
- zengs_gamma.py
- T_ET_metric.py
- betts_approach.py
- two_legged_metric.py

Script providing examples of applying each diagnostic:
- examples_of_application.py

These scripts are largely based on the Iris Python package (https://scitools.org.uk/iris/docs/latest/) and are designed to be run with observations and model output in NetCDF format.

If you use any of these scripts in your research please cite the reference above. If you have any questions, comments or feedback please email Jess Baker using the email address at the top of this document. 
