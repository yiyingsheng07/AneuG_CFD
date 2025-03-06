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

### Use AneuG to get CFD meshes for generated shapes

1. Clone AneuG into VMTK directory

```bash
cd VMTK
git clone https://github.com/anonymousaneug/AneuG.git
```

2. Follow instructions at https://github.com/anonymousaneug/AneuG.git and download network checkpoint files from Google drive.

3. Run script to generate synthetic shapes:
```bash
cd AneuG
python pipeline_generator.py
```

4. Run script to create volume mesh using VMTK:
```bash
cd ..
python pipeline_remesher.py
```

5 Run script to sort out inlet & outlet sequence for generated .vtu files:
```bash
cd ..
python pipeline_sort_parts.py
```
