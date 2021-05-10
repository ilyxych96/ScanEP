# ScanEP


ScanEP is a multimodule programm that helps to calculate energy profiles. It include next modules:
  - VEC Changer
  - Method changer
  - Inp files generator
  - Molecular delimiter
  - QDPT result processing

## Module assignment and Instructions
### VEC Changer
>If you need write VEC group to all files in folder you can:
  - Set as "main file" path to folder with .inp files
  - Set as "vec file" path to file contain VEC group

### Method changer
>If you need change calculation method for all files in folder you can:
  - Set as "Directory with files to change" path to folder with .inp files
  - Set as "New method file" path to file contain New method

### Inp files generator

> Main module allows interpolating geometry structure in intramolecular coordinates between two states in Cartesian coordinates. More information you can find in [].
  - Set as "Settings" path to folder with .inp files
  - Set as "Geometry 1" path to file with first geometry in xyz format
  - Set as "Geometry 2" path to file with second geometry in xyz format 
  - Set as "Directory with files to change" path to directory where will be generating new files
  - Type in "inp filenames mask" name mask for new files
  - Select "dimer/monomer" (depends on the subject of your research)
  - Set interpolation step (Sometimes we need increase accuracy of EP so we need intermediate structures, so I recommend use step = 0.01 and then use only needed files)

### Molecular delimiter
> If you have molecular modeling cell or crystall fragment and you need monomers and dimers you can use this module
  - Set as "Pack file" path to file with atoms coordinates in xyz format
  - Type in "Minimum atoms in one molecule" minimum size of molecule (it is filter for small molecules)
  - Type in "Maximum contact length" maximum contact length between molecues (it is filter for amount of output dimers)  

### QDPT result processing
>This module can vizualize calculating energies of two states
  - Set as "Directory with QDPT files" path to folder with QDPT .out files
  - Set lines and grid settings

## Installation

Download all files in directory .../ScanEp/

The following libraries are required to run:
>PqQT5, os, re, glob, shutil, matplotlib, numpy, math, copy, time.

If you have all this packages, run ScanEp.pyw

## Todos

 - MORE Tests
 - Upgrade z-matrix builder for 3+ molecules

License
----

MIT

**Free Software, Enjoy!**
***Have questions? Write me: scanepfeedback@gmail.com*** 
