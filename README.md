# AneuG_CFD
This repository provides an assembled automatic pipeline to create a large amount of 3D volume meshes ready for Fluent simulations. This includes 3 parts:
- I. Generating synthetic intracranial aneurysm shapes using [AneuG](https://github.com/anonymousaneug/AneuG.git).
- II. Surface smoothing and fixing of imperfections (automation script of Geomagic). 
- III. Creating 3D volume mesh from the refined surface mesh (VMTK).

### Acknowledgement
Most of the code in the subdir ```VMTK``` is adapted from [CFD_Machine_Learning/VMTK](https://github.com/EndritPJ/CFD_Machine_Learning/tree/main/VMTK).


## Mesh Smoothing
We provide a automation script of Geomagic Wrap to conduct mesh post-processing. This elminates a series of mesh defects, remesh the shape to obtain higher-resolution, removes unsmooth local spikes, and improves volume meshing success rate. See:
```bash
Geomagic/automatic_mesh_fixing.txt
```
(I haven't put it in the repo)

## Createing 3D Volume Mesh for AneuG-generated Shapes
If you wish to conduct 3D volume meshing of AneuG-generated shapes, run script:
```bash
cd VMTK
python get_mesh_dataset_custom.py
```
It is to be noted that one files in each subfolder is necessary if you wish to directly run the script without any modification. This file is called "checkpoint.npy" which contains many different kinds of shape information. In simple words, this checkpoint file contains the point cloud of parent vessel centrelines, which helps the function [`sort_parts`](VMTK/get_mesh_dataset_custom.py) to figure out which opening is the inlet and which are outlets. This would ensure the volume meshes for your downstream application equip the same configuration of inlet and outlets so automation can be achieved for the CFD part.
A function is provided to create this checkpoint file. You can adapt the function [`get_mesh_fusion_dataset`](VMTK/AneuG/visualization/gallery.py) for your task.


## More about The Volume Meshing Script
If you wish to get volume meshes for your own shapes, you would need to adapt the script [`VMTK/get_mesh_dataset_custom.py`](VMTK/get_mesh_dataset_custom.py). Arguements controlling the meshing size are injected into the function [`cfdmesher_custom`](VMTK/vmtk_cfdmesher.py).



