gs2_3d
======

This simple program reads on a GS2 NetCDF output file and calculates the real
space coordinates and exports them to useful file formats such as CSV or VTK.
It does not plot the flux tube since Matplotlib is in general not appropriate
for visualizing hundreds of thousands of points in 3D.

CSV
---

Setting *write_fields = csv* in the config file will write a CSV file giving
with one data point per line, i.e. in the form (X, Y, Z, F) where F is the
value of the field at that point. This can then be loaded into other plotting
programs such as Mayavi in Python and plotted as a scatter plot. Matplotlib
could probably produce static plots, but will probably be very slow for any
dynamic manipulation!

VTK
---

The VTK format, and associated visualization applications such Paraview, are
ideally suited to visualizing large numbers of data points. There are two types
of VTK files: legacy and XML files. Legacy files are easily created text files
whereas XML VTK files support random access, parallel I/O, and portable data
compression. *gs2_3d* writes legacy vtk files since writing XML files creates
additional dependencies. This feature is not planned but is also not hard to
implement.
