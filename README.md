# AneuG_CFD

### Pre-processing Pipeline
*The code in the ```VMTK``` folder is copied from [CFD_Machine_Learning/VMTK](https://github.com/EndritPJ/CFD_Machine_Learning/tree/main/VMTK). It has not been modified unless stated otherwise.

1. .obj to .vtp file (x)

   Paraview save as .vtp

2. Smooth surface (x)

   vmtksurfacesmoothing -ifile filename.vtp -passband 0.1 -iterations 30 -ofile filename_sm.vtp

3. Flow extension (optional)

   vmtksurfacereader -ifile filename_sm.vtp --pipe vmtkcenterlines -seedselector openprofiles --pipe vmtkflowextensions -adaptivelength 1 -extensionratio 5 -normalestimationratio 1 -interactive 0 --pipe vmtksurfacewriter -ofile filename_sm_ex.vtp
   (for more than one outlet, input both ids with blank space between them)

4. Mesh generation (code from [CFD_Machine_Learning/VMTK](https://github.com/EndritPJ/CFD_Machine_Learning/tree/main/VMTK))

   ```vmtk_cfdmesher.py```

5. Assign inlet and outlet (in progress)
6. .vtu to .msh for Fluent
