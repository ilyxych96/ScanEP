ScanEP it is multi-module GUI program
The main goal is linear iterpolation between two xyz format geometries using intermolecular coordinates.
To do this I perform the transformation from Cartesian coordinates to Z-matrix, linear interpolation in internal coordinates and finally reversed transformation from Z-matrix to Cartesian coordinates
As a result we get two sets of files: set of .inp files in GAMESS/Firefly format and .xyz files with intermediate coorinates.
The second main block named "Molecular Delimiter" can split a molecular dynamic cell or crystal fragment into monomers and molecular pairs and sorts them in ascending order of shortest contact distance
Module "QDPT result processing" plot energy profile. Enter path to folder with QDPT result and you will see EP. 
