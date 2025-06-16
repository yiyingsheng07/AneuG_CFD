from vmtk_cfdmesher import cfdmesher_single, cfdmesher_custom
import os, readline
import vmtk
from vmtk import vmtkscripts
import numpy as np
import vtk
from vtk.vtkCommonCore import vtkObject
from vtk.util import numpy_support as ns
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from vmtk_vtu2msh import write_msh_single
import pandas as pd
import pyvista as pv
import random
from get_mesh_dataset_custom import scan_inlet_nodes


def sort_parts(mesh_file, visualize=False, skip=False):
    # read vtu
    mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
    mesh_reader.InputFileName = mesh_file
    mesh_reader.Execute()
    # convert vtu to np array
    mesh2np = vmtk.vmtkmeshtonumpy.vmtkMeshToNumpy()
    mesh2np.Mesh = mesh_reader.Mesh
    mesh2np.Execute()
    mesh_arr = mesh2np.ArrayDict
    # read vtu with vtk
    vtk_reader = vtk.vtkXMLUnstructuredGridReader()
    vtk_reader.SetFileName(mesh_file)
    vtk_reader.Update()
    ugrid = vtk_reader.GetOutput()
    total_cell_number = ugrid.GetNumberOfCells()
    if not skip:
        cell_ids = np.unique(mesh_arr['CellData']['CellEntityIds'])
        cell_number = [len(np.where(mesh_arr['CellData']['CellEntityIds']==cell_id)[0]) for cell_id in cell_ids]
        
        # load opening checkpoint
        checkpoint_path = os.path.join(os.path.dirname(mesh_file), "checkpoint.npy")
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        cpcd_glo_gen = checkpoint["cpcd_glo_gen"] * checkpoint['norm_canonical']
        # cpcd_glo_gen = cpcd_glo_gen[:, -1:, :]  # [num_branch, 1, 3]

        id_set = [2, 3, 4]
        # get sort sequence
        cell_coordinates = []
        for cell_type in id_set:
            cell_id = np.where(mesh_arr['CellData']['CellEntityIds']==cell_type)[0][0]
            cell_coord = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_id).GetPoints().GetData()))
            cell_coordinates.append(cell_coord) 
        cell_coordinates = np.stack(cell_coordinates, axis=0).mean(axis=1)  # [num_branch, 3]
        cell_coordinates = np.expand_dims(cell_coordinates, axis=0)  # [1, num_branch, 3]
        distances = np.linalg.norm(cell_coordinates - cpcd_glo_gen[:, -1:, :], axis=-1)  # [num_branch, num_branch]
        sort_sequence = np.argmin(distances, axis=0)

        # sort parts
        point_set = {}
        for i in id_set:
            point_set[str(i)] = []
        for i, cell_id in enumerate(mesh_arr['CellData']['CellEntityIds']):
                if cell_id in id_set:
                        original_id = id_set.index(cell_id)
                        new_id = id_set[sort_sequence[original_id]]
                        mesh_arr['CellData']['CellEntityIds'][i] = new_id
        
        # overwrite
        np2mesh = vmtk.vmtknumpytomesh.vmtkNumpyToMesh()
        np2mesh.ArrayDict = mesh_arr
        np2mesh.Execute()
        writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
        writer.Mesh = np2mesh.Mesh
        writer.OutputFileName = mesh_file
        writer.Execute()


        # record points
        for cell_id in id_set:
            cell_indices = np.where(mesh_arr['CellData']['CellEntityIds']==cell_id)[0]
            for cell_indice in cell_indices:
                points = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_indice).GetPoints().GetData()))
                point_set[str(cell_id)].append(points)
            point_set[str(cell_id)] = np.concatenate(point_set[str(cell_id)], axis=0)

        # visualize
        if visualize:
            figure = plt.figure()
            ax = figure.add_subplot(111, projection='3d')
            colors = ['r', 'g', 'b']
            for i, color in zip(id_set, colors):
                points = point_set[str(i)]
                cpcd_points = cpcd_glo_gen.reshape(-1, 3)
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)
                ax.scatter(cpcd_points[:, 0], cpcd_points[:, 1], cpcd_points[:, 2], c='gray')
            plt.savefig(os.path.join(os.path.dirname(mesh_file), "sorted_parts.png"))
    return total_cell_number

if __name__ == "__main__":
    # conf
    # root = os.path.join(os.getcwd(), "AneuG_CFD/stable_64_v1" ) # change this to relative path on your workstation
    root = "/media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence"
    case_list = [7089, 9900]
    dir_list = [os.path.join(root, str(case)) for case in case_list]
    dpi = 12
    edge_list  = [0.24, 0.22, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.135, 0.13, 0.12, 0.11]
    max_edge_list = [1.0, 1.0, 1.0, 1.0, 1.0]
    inflation = "y"

    readline.set_completer_delims(" \t\n=")
    readline.parse_and_bind("tab: complete")
    
    sub_dir_list = []

    for dir in dir_list:
        for i in range(dpi):
            sub_dir = os.path.join(dir, str(i))
            sub_dir_list.append(sub_dir)
    sub_dir_list_shuffled = random.sample(sub_dir_list, len(sub_dir_list))
    for sub_dir in sub_dir_list_shuffled:
        i = int(sub_dir.split("/")[-1])
        src = os.path.join(sub_dir, "shape_remeshed.vtp")
        vtu_path = os.path.join(sub_dir, "mesh.vtu")
        msh_path = os.path.join(sub_dir, "mesh.msh")
        if not os.path.exists(msh_path):
            print("Mesh file not found, generating mesh... for {}".format(sub_dir))
            if not os.path.exists(vtu_path):
                cfdmesher_custom(src, vtu_path, edge_list[i], 1, inflation)
            total_cell_number = sort_parts(mesh_file=vtu_path)
            log_str = "Case {}, DPI {}, Total Cell Number: {}".format(dir, i, total_cell_number)
            print(log_str)
            with open(os.path.join(root, "cell_counts.txt"), "a") as f:
                f.write(log_str + "\n")
            scan_inlet_nodes(vtu_path)
            if not os.path.exists(msh_path):
                write_msh_single(ifile=vtu_path, ofile=msh_path)

    cell_count = []
    for sub_dir in sub_dir_list:
        msh_path = os.path.join(sub_dir, "mesh.msh")
        vtu_path = os.path.join(sub_dir, "mesh.vtu")
        total_cell_number = sort_parts(mesh_file=vtu_path, skip=True)
        i = int(sub_dir.split("/")[-1])
        print("Case {}, DPI {}, Total Cell Number: {}".format(sub_dir, i, total_cell_number))       
        cell_count.append(round(total_cell_number/10000, 1))
    print(cell_count)
"""
cd VMTK
conda activate vmtk_add
python mesh_convergence.py


rm -r /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence



rm -r /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence/4486
rm -r /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence/6584

cd /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence/5120

4486 0, 1, 3
6584 3, 6


find . -type f -name "*.msh" -print0 | while IFS= read -r -d '' file; do
    echo "$file"
done


find . -type f -name "ensight_files1600.vel" -print0 | while IFS= read -r -d '' file; do
    echo "$file"
done
[5120, 7089, 9900]

cd /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence/5120

find . -type f -name "mesh.msh" | wc -l

cd /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence/7089
find . -type f -name "mesh.msh" | wc -l


rsync -r --progress /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/mesh_convergence wenhao@100.101.90.86:/media/yaplab/HDD_Storage/wenhao/datasets/AneuG_CFD

"""