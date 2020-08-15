import csv
import os
import re
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
import math as m
import copy
import time


class Converter:
    """A coordinate converter class"""

    def __init__(self):
        # Dictionary of the masses of elements indexed by element name;
        # includes X for dummy atoms
        self.elements = {'Ac': 89, 'Al': 13, 'Am': 95, 'Sb': 51, 'Ar': 18, 'As': 33, 'At': 85, 'Ba': 56, 'Bk': 97,
                         'Be': 4, 'Bi': 83, 'Bh': 107, 'B': 5, 'Br': 35, 'Cd': 48, 'Ca': 20, 'Cf': 98, 'C': 6,
                         'Ce': 58, 'Cs': 55, 'Cl': 17, 'Cr': 24, 'Co': 27, 'Cu': 29, 'Cm': 96, 'Db': 105, 'Dy': 66,
                         'Es': 99, 'Er': 68, 'Eu': 63, 'Fm': 100, 'F': 9, 'Fr': 87, 'Gd': 64, 'Ga': 31, 'Ge': 32,
                         'Au': 79, 'Hf': 72, 'Hs': 108, 'He': 2, 'Ho': 67, 'H': 1, 'In': 49, 'I': 53, 'Ir': 77,
                         'Fe': 26, 'Kr': 36, 'La': 57, 'Lr': 103, 'Pb': 82, 'Li': 3, 'Lu': 71, 'Mg': 12, 'Mn': 25,
                         'Mt': 109, 'Md': 101, 'Hg': 80, 'Mo': 42, 'Nd': 60, 'Ne': 10, 'Np': 93, 'Ni': 28, 'Nb': 41,
                         'N': 7, 'No': 102, 'Os': 76, 'O': 8, 'Pd': 46, 'P': 15, 'Pt': 78, 'Pu': 94, 'Po': 84,
                         'K': 19, 'Pr': 59, 'Pm': 61, 'Pa': 91, 'Ra': 88, 'Rn': 86, 'Re': 75, 'Rh': 45, 'Rb': 37,
                         'Ru': 44, 'Rf': 104, 'Sm': 62, 'Sc': 21, 'Sg': 106, 'Se': 34, 'Si': 14, 'Ag': 47, 'Na': 11,
                         'Sr': 38, 'S': 16, 'Ta': 73, 'Tc': 43, 'Te': 52, 'Tb': 65, 'Tl': 81, 'Th': 90, 'Tm': 69,
                         'Sn': 50, 'Ti': 22, 'W': 74, 'U': 92, 'V': 23, 'Xe': 54, 'Yb': 70, 'Y': 39, 'Zn': 30,
                         'Zr': 40}
        self.masses = {'X': 0, 'Ac': 227.028, 'Al': 26.981539, 'Am': 243, 'Sb': 121.757, 'Ar': 39.948,
                       'As': 74.92159, 'At': 210, 'Ba': 137.327, 'Bk': 247, 'Be': 9.012182, 'Bi': 208.98037,
                       'Bh': 262, 'B': 10.811, 'Br': 79.904, 'Cd': 112.411, 'Ca': 40.078, 'Cf': 251, 'C': 12.011,
                       'Ce': 140.115, 'Cs': 132.90543, 'Cl': 35.4527, 'Cr': 51.9961, 'Co': 58.9332, 'Cu': 63.546,
                       'Cm': 247, 'Db': 262, 'Dy': 162.5, 'Es': 252, 'Er': 167.26, 'Eu': 151.965, 'Fm': 257,
                       'F': 18.9984032, 'Fr': 223, 'Gd': 157.25, 'Ga': 69.723, 'Ge': 72.61, 'Au': 196.96654,
                       'Hf': 178.49, 'Hs': 265, 'He': 4.002602, 'Ho': 164.93032, 'H': 1.00794, 'In': 114.82,
                       'I': 126.90447, 'Ir': 192.22, 'Fe': 55.847, 'Kr': 83.8, 'La': 138.9055, 'Lr': 262,
                       'Pb': 207.2, 'Li': 6.941, 'Lu': 174.967, 'Mg': 24.305, 'Mn': 54.93805,
                       'Mt': 266, 'Md': 258, 'Hg': 200.59, 'Mo': 95.94, 'Nd': 144.24, 'Ne': 20.1797, 'Np': 237.048,
                       'Ni': 58.6934, 'Nb': 92.90638, 'N': 14.00674, 'No': 259, 'Os': 190.2, 'O': 15.9994,
                       'Pd': 106.42, 'P': 30.973762, 'Pt': 195.08, 'Pu': 244, 'Po': 209, 'K': 39.0983,
                       'Pr': 140.90765, 'Pm': 145, 'Pa': 231.0359, 'Ra': 226.025, 'Rn': 222, 'Re': 186.207,
                       'Rh': 102.9055, 'Rb': 85.4678, 'Ru': 101.07, 'Rf': 261, 'Sm': 150.36, 'Sc': 44.95591,
                       'Sg': 263, 'Se': 78.96, 'Si': 28.0855, 'Ag': 107.8682, 'Na': 22.989768, 'Sr': 87.62,
                       'S': 32.066, 'Ta': 180.9479, 'Tc': 98, 'Te': 127.6, 'Tb': 158.92534, 'Tl': 204.3833,
                       'Th': 232.0381, 'Tm': 168.93421, 'Sn': 118.71, 'Ti': 47.88, 'W': 183.85, 'U': 238.0289,
                       'V': 50.9415, 'Xe': 131.29, 'Yb': 173.04, 'Y': 88.90585, 'Zn': 65.39, 'Zr': 91.224}
        self.total_mass = 0
        self.cartesian = []
        self.new_order = []
        self.zmatrix = []

    def read_zmatrix(self, input_file):
        """
        Read the input zmatrix file (assumes no errors and no variables)
        The zmatrix is a list with each element formatted as follows
        [ name, [[ atom1, distance ], [ atom2, angle ], [ atom3, dihedral ]], mass ]
        The first three atoms have blank lists for the undefined coordinates
        """
        self.zmatrix = []
        with open(input_file, 'r') as f:
            #f.readline()
            #f.readline()
            name = f.readline().strip()
            self.zmatrix.append([name, [], self.masses[name]])
            name, atom1, distance = f.readline().split()[:3]
            self.zmatrix.append([name,
                                 [[int(atom1) - 1, float(distance)], [], []],
                                 self.masses[name]])
            name, atom1, distance, atom2, angle = f.readline().split()[:5]
            self.zmatrix.append([name,
                                 [[int(atom1) - 1, float(distance)],
                                  [int(atom2) - 1, np.radians(float(angle))], []],
                                 self.masses[name]])
            for line in f.readlines():
                # Get the components of each line, dropping anything extra
                name, atom1, distance, atom2, angle, atom3, dihedral = line.split()[:7]
                # convert to a base 0 indexing system and use radians
                atom = [name,
                        [[int(atom1) - 1, float(distance)],
                         [int(atom2) - 1, np.radians(float(angle))],
                         [int(atom3) - 1, np.radians(float(dihedral))]],
                        self.masses[name]]
                self.zmatrix.append(atom)

        return self.zmatrix

    def read_cartesian(self, count, input_file) :
        """
        Read the cartesian coordinates file (assumes no errors)
        The cartesian coordiantes consist of a list of atoms formatted as follows
        [ name, np.array( [ x, y, z ] ), mass ]
        """
        num2nam = {89 : 'Ac', 13 : 'Al', 95 : 'Am', 51 : 'Sb', 18 : 'Ar', 33 : 'As', 85 : 'At', 56 : 'Ba', 97 : 'Bk',
                   4 : 'Be',
                   83 : 'Bi', 107 : 'Bh', 5 : 'B', 35 : 'Br', 48 : 'Cd', 20 : 'Ca', 98 : 'Cf', 6 : 'C', 58 : 'Ce',
                   55 : 'Cs',
                   17 : 'Cl', 24 : 'Cr', 27 : 'Co', 29 : 'Cu', 96 : 'Cm', 105 : 'Db', 66 : 'Dy', 99 : 'Es', 68 : 'Er',
                   63 : 'Eu',
                   100 : 'Fm', 9 : 'F', 87 : 'Fr', 64 : 'Gd', 31 : 'Ga', 32 : 'Ge', 79 : 'Au', 72 : 'Hf', 108 : 'Hs',
                   2 : 'He',
                   67 : 'Ho', 1 : 'H', 49 : 'In', 53 : 'I', 77 : 'Ir', 26 : 'Fe', 36 : 'Kr', 57 : 'La', 103 : 'Lr',
                   82 : 'Pb',
                   3 : 'Li', 71 : 'Lu', 12 : 'Mg', 25 : 'Mn', 109 : 'Mt', 101 : 'Md', 80 : 'Hg', 42 : 'Mo', 60 : 'Nd',
                   10 : 'Ne',
                   93 : 'Np', 28 : 'Ni', 41 : 'Nb', 7 : 'N', 102 : 'No', 76 : 'Os', 8 : 'O', 46 : 'Pd', 15 : 'P',
                   78 : 'Pt',
                   94 : 'Pu', 84 : 'Po', 19 : 'K', 59 : 'Pr', 61 : 'Pm', 91 : 'Pa', 88 : 'Ra', 86 : 'Rn', 75 : 'Re',
                   45 : 'Rh',
                   37 : 'Rb', 44 : 'Ru', 104 : 'Rf', 62 : 'Sm', 21 : 'Sc', 106 : 'Sg', 34 : 'Se', 14 : 'Si', 47 : 'Ag',
                   11 : 'Na',
                   38 : 'Sr', 16 : 'S', 73 : 'Ta', 43 : 'Tc', 52 : 'Te', 65 : 'Tb', 81 : 'Tl', 90 : 'Th', 69 : 'Tm',
                   50 : 'Sn',
                   22 : 'Ti', 74 : 'W', 92 : 'U', 23 : 'V', 54 : 'Xe', 70 : 'Yb', 39 : 'Y', 30 : 'Zn', 40 : 'Zr'}

        # реверсивный словарь
        # nam2num = dict(zip(num2nam.values(), num2nam.keys()))

        self.cartesian = []
        indexes = []
        with open(input_file, 'r') as f :
            f.readline()
            f.readline()
            if f.readline()[0].isdigit() :
                f.seek(0)
                i = 1
                for line in f.readlines() :
                    name, x, y, z = line.split()
                    name = num2nam[int(name)]
                    self.cartesian.append([name, np.array([x, y, z], dtype='f8'), self.masses[name]])
                    indexes.append(i)
                    i = i + 1
            else :
                f.seek(0)
                f.readline()
                f.readline()
                i = 1
                for line in f.readlines() :
                    name, x, y, z = line.split()
                    self.cartesian.append([name, np.array([x, y, z], dtype='f8'), self.masses[name]])
                    indexes.append(i)
                    i = i + 1

        '''if count == 0:
            self.new_order = self.closest2cm()
            new_cartesian = []
            for item in self.new_order :
                new_cartesian.append(self.cartesian[item - 1])

            self.cartesian = new_cartesian

            f = open('zzzzzzzz.xyz', 'w')
            line = self.str_cartesian()
            f.write(line)
            f.close()
        else:
            new_cartesian = []
            for item in self.new_order :
                new_cartesian.append(self.cartesian[item - 1])

            self.cartesian = new_cartesian

            f = open('zzzzzzzz.xyz', 'w')
            line = self.str_cartesian()
            f.write(line)
            f.close()'''

        return self.cartesian

    def rotation_matrix(self, axis, angle):
        """
        Euler-Rodrigues formula for rotation matrix
        """
        # Normalize the axis
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)
        return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

    def add_first_three_to_cartesian(self):
        """
        The first three atoms in the zmatrix need to be treated differently
        """
        # First atom
        name, coords, mass = self.zmatrix[0]
        self.cartesian = [[name, np.array([0., 0., 0.]), mass]]

        # Second atom
        name, coords, mass = self.zmatrix[1]
        distance = coords[0][1]
        self.cartesian.append(
            [name, np.array([distance, 0., 0.]), self.masses[name]])

        # Third atom
        name, coords, mass = self.zmatrix[2]
        atom1, atom2 = coords[:2]
        atom1, distance = atom1
        atom2, angle = atom2
        q = np.array(self.cartesian[atom1][1], dtype='f8')  # position of atom 1
        r = np.array(self.cartesian[atom2][1], dtype='f8')  # position of atom 2

        # Vector pointing from q to r
        a = r - q

        # Vector of length distance pointing along the x-axis
        d = distance * a / np.sqrt(np.dot(a, a))

        # Rotate d by the angle around the z-axis
        d = np.dot(self.rotation_matrix([0, 0, 1], angle), d)

        # Add d to the position of q to get the new coordinates of the atom
        p = q + d
        atom = [name, p, self.masses[name]]
        self.cartesian.append(atom)

    def add_atom_to_cartesian(self, coords):
        """Find the cartesian coordinates of the atom"""
        name, coords, mass = coords
        atom1, distance = coords[0]
        atom2, angle = coords[1]
        atom3, dihedral = coords[2]

        q = self.cartesian[atom1][1]  # atom 1
        r = self.cartesian[atom2][1]  # atom 2
        s = self.cartesian[atom3][1]  # atom 3

        # Vector pointing from q to r
        a = r - q
        # Vector pointing from s to r
        b = r - s

        # Vector of length distance pointing from q to r
        d = distance * a / np.sqrt(np.dot(a, a))

        # Vector normal to plane defined by q, r, s
        normal = np.cross(a, b)
        # Rotate d by the angle around the normal to the plane defined by q, r, s
        d = np.dot(self.rotation_matrix(normal, angle), d)

        # Rotate d around a by the dihedral
        d = np.dot(self.rotation_matrix(a, dihedral), d)

        # Add d to the position of q to get the new coordinates of the atom
        p = q + d
        atom = [name, p, mass]

        self.cartesian.append(atom)

    def zmatrix_to_cartesian(self):
        """
        Convert the zmartix to cartesian coordinates
        """
        # Deal with first three line separately
        self.add_first_three_to_cartesian()

        for atom in self.zmatrix[3:]:
            self.add_atom_to_cartesian(atom)

        self.remove_dummy_atoms()

        self.center_cartesian()

        return self.cartesian

    def add_first_three_to_zmatrix(self):
        """The first three atoms need to be treated differently"""
        # First atom
        self.zmatrix = []
        name, position, mass = self.cartesian[0]
        self.zmatrix.append([name, [[], [], []], mass])

        # Second atom
        if len(self.cartesian) > 1:
            name, position, mass = self.cartesian[1]
            atom1 = self.cartesian[0]
            pos1 = atom1[1]
            q = pos1 - position
            distance = np.sqrt(np.dot(q, q))
            self.zmatrix.append([name, [[0, distance], [], []], mass])

        # Third atom
        if len(self.cartesian) > 2:
            name, position, mass = self.cartesian[2]
            atom1, atom2 = self.cartesian[:2]
            pos1, pos2 = atom1[1], atom2[1]
            q = pos1 - position
            r = pos2 - pos1
            q_u = q / np.sqrt(np.dot(q, q))
            r_u = r / np.sqrt(np.dot(r, r))
            distance = np.sqrt(np.dot(q, q))
            # Angle between a and b = acos(dot(a, b)) / (|a| |b|))
            angle = np.arccos(np.dot(-q_u, r_u))
            self.zmatrix.append(
                [name, [[0, distance], [1, np.degrees(angle)], []], mass])

    def add_atom_to_zmatrix(self, i, line):
        """Generates an atom for the zmatrix
        (assumes that three previous atoms have been placed in the cartesian coordiantes)"""
        name, position, mass = line
        atom1, atom2, atom3 = self.cartesian[i-3:i]
        pos1, pos2, pos3 = atom1[1], atom2[1], atom3[1]
        # Create vectors pointing from one atom to the next
        q = pos1 - position
        r = pos2 - pos1
        s = pos3 - pos2
        position_u = position / np.sqrt(np.dot(position, position))
        # Create unit vectors
        q_u = q / np.sqrt(np.dot(q, q))
        r_u = r / np.sqrt(np.dot(r, r))
        s_u = s / np.sqrt(np.dot(s, s))
        distance = np.sqrt(np.dot(q, q))
        # Angle between a and b = acos(dot(a, b)) / (|a| |b|))
        angle = np.arccos(np.dot(-q_u, r_u))
        angle_123 = np.arccos(np.dot(-r_u, s_u))
        # Dihedral angle =
        #   acos(dot(normal_vec1, normal_vec2)) / (|normal_vec1| |normal_vec2|))
        plane1 = np.cross(q, r)
        plane2 = np.cross(r, s)
        dihedral = np.arccos(np.dot(
            plane1, plane2) / (np.sqrt(np.dot(plane1, plane1)) * np.sqrt(np.dot(plane2, plane2))))
        # Convert to signed dihedral angle
        if np.dot(np.cross(plane1, plane2), r_u) < 0:
            dihedral = -dihedral

        coords = [[0, distance], [1, np.degrees(angle)], [
            2, np.degrees(dihedral)]]
        atom = [name, coords, mass]
        self.zmatrix.append(atom)

    def cartesian_to_zmatrix(self):
        """Convert the cartesian coordinates to a zmatrix"""
        self.add_first_three_to_zmatrix()
        for i, atom in enumerate(self.cartesian[3:], start=3):
            self.add_atom_to_zmatrix(i, atom)

        return self.zmatrix

    def remove_dummy_atoms(self):
        """Delete any dummy atoms that may have been placed in the calculated cartesian coordinates"""
        new_cartesian = []
        for atom, xyz, mass in self.cartesian:
            if not atom == 'X':
                new_cartesian.append((atom, xyz, mass))
        self.cartesian = new_cartesian

    def center_cartesian(self):
        """Find the center of mass and move it to the origin"""
        self.total_mass = 0.0
        center_of_mass = np.array([0.0, 0.0, 0.0])
        for atom, xyz, mass in self.cartesian:
            self.total_mass += mass
            center_of_mass += xyz * mass
        center_of_mass = center_of_mass / self.total_mass

        # Translate each atom by the center of mass
        for atom, xyz, mass in self.cartesian:
            xyz -= center_of_mass

    def cartesian_radians_to_degrees(self):
        for atom in self.cartesian:
            atom[1][1][1] = np.degrees(atom[1][1][1])
            atom[1][2][1] = np.degrees(atom[1][2][1])

    def output_cartesian(self, output_file='cartesian.xyz'):
        """Output the cartesian coordinates of the file"""
        with open(output_file, 'w') as f:
            f.write(f'{len(self.cartesian)}\n\n')
            f.write(self.str_cartesian())

    def str_cartesian(self):
        """Print the cartesian coordinates"""
        out = ''
        for atom, (x, y, z), masses in self.cartesian:
            out += f'{atom:<2s} {x:>15.10f} {y:>15.10f} {z:>15.10f}\n'

        return out

    def output_zmatrix(self, output_file='zmatrix.dat'):
        """Output the zmatrix to the file"""
        with open(output_file, 'w') as f:
            f.write(self.str_zmatrix())

    def str_zmatrix(self):
        """Print the zmatrix"""
        out = f'{self.zmatrix[0][0]}\n'
        counter = 0
        for atom, position, mass in self.zmatrix[1:]:
            out += f'{atom:<2s}'
            counter += 1
            for i in position:
                for j in range(0, len(i), 2):
                    out += f' {i[j] + 1:>3d} {i[j + 1]:>15.10f}'
            out += '\n'

        return out

    def run_zmatrix(self, input_file, output_file):
        """Read in the zmatrix, converts it to cartesian, and outputs it to a file"""
        self.read_zmatrix(input_file)
        self.zmatrix_to_cartesian()
        self.output_cartesian(output_file)

    def run_cartesian(self, count, input_file, output_file):
        """Read in the cartesian coordinates, convert to cartesian, and output the file"""
        self.read_cartesian(count, input_file)
        self.cartesian_to_zmatrix()
        self.output_zmatrix(output_file)

    def closest2cm(self):
        """Find the center of mass and move it to the origin"""
        self.total_mass = 0.0
        self.sorted_atoms = []
        center_of_mass = np.array([0.0, 0.0, 0.0])

        for atom, xyz, mass in self.cartesian :
            self.total_mass += mass
            center_of_mass += xyz * mass
        center_of_mass = center_of_mass / self.total_mass
        print(center_of_mass)

        length = []
        atoms = []
        i = 1
        for atom, xyz, mass in self.cartesian:
            length.append(m.sqrt(pow((xyz[0]-center_of_mass[0]), 2) + pow(xyz[1]-center_of_mass[1], 2) + pow(xyz[2]-center_of_mass[2], 2)))
            atoms.append(i)
            i += 1

        originlist = dict(zip(length, atoms))
        length.sort()
        filteredlist = {}
        for i in length:
            filteredlist[i] = originlist[i]

        sorted_atoms = list(filteredlist.values())

        return sorted_atoms


def zmatrixgenerator(xyz1, xyz2, outfile1, outfile2, atomsinfirst):

    def read_cartesian(file):
        xyz = open(file, 'r')
        cartesian = []
        atomnumber = 1
        xyz.readline()
        xyz.readline()
        if xyz.readline()[0].isdigit() :
            xyz.seek(0)
            for line in xyz :
                atom, x, y, z = line.split()
                cartesian.append([num2nam[int(atom)], np.array([x, y, z], dtype='f8'), atomnumber])
                atomnumber = atomnumber + 1
        else :
            xyz.seek(0)
            xyz.readline()
            xyz.readline()
            for line in xyz:
                atom, x, y, z = line.split()
                cartesian.append([atom, np.array([x, y, z], dtype='f8'), atomnumber])
                atomnumber = atomnumber + 1

        xyz.close()
        return cartesian

    # output: 0)atom / 1)xyz / 2)length / 3)index in cartesian list / 4)index in zmatrix list / 5) find index / 6) origin index
    def closestatom_cartesian(atoms, current_atom, count) :
        length = 1000
        i = 0
        for atom, xyz, atomnumber in atoms :
            currentlength = (m.sqrt(pow((float(xyz[0]) - float(current_atom[1][0])), 2) +
                                    pow((float(xyz[1]) - float(current_atom[1][1])), 2) +
                                    pow((float(xyz[2]) - float(current_atom[1][2])), 2)))
            if currentlength < length :
                atomout = atom
                xyzout = xyz
                index = i
                length = currentlength
                atomnumberout = atomnumber
            i = i + 1

        out = [atomout, xyzout, length, index, count, count, atomnumberout]
        return out

    def closestatom_using(atoms, current_atom) :
        length = 1000
        i = 0
        for atom, xyz, lengthtmp, index, count, temp, atomnumber in atoms :
            currentlength = (m.sqrt(pow((float(xyz[0]) - float(current_atom[1][0])), 2) +
                                    pow((float(xyz[1]) - float(current_atom[1][1])), 2) +
                                    pow((float(xyz[2]) - float(current_atom[1][2])), 2)))
            if currentlength <= length :
                atomout = atom
                xyzout = xyz
                length = currentlength
                indexout = index
                countout = count
                current_index = i
                atomnumberout = atomnumber
            i = i + 1

        out = [atomout, xyzout, length, indexout, countout, current_index, atomnumberout]
        return out

    def lengthcalc(atom1, atom2):
        xyz1 = atom1[1]
        xyz2 = atom2[1]
        currentlength = (m.sqrt(pow((float(xyz1[0]) - float(xyz2[0])), 2) +
                                pow((float(xyz1[1]) - float(xyz2[1])), 2) +
                                pow((float(xyz1[2]) - float(xyz2[2])), 2)))
        return currentlength

    def angle(atom1, atom2, atom3):
        xyz1 = atom1[1]
        xyz2 = atom2[1]
        xyz3 = atom3[1]
        q = xyz2 - xyz1
        r = xyz3 - xyz2
        q_u = q / np.sqrt(np.dot(q, q))
        r_u = r / np.sqrt(np.dot(r, r))
        # Angle between a and b = acos(dot(a, b)) / (|a| |b|))
        angle = np.degrees(np.arccos(np.dot(-q_u, r_u)))
        return angle

    def dihedral(atom1, atom2, atom3, atom4):
        xyz1 = atom1[1]
        xyz2 = atom2[1]
        xyz3 = atom3[1]
        xyz4 = atom4[1]
        q = xyz2 - xyz1
        r = xyz3 - xyz2
        s = xyz4 - xyz3
        r_u = r / np.sqrt(np.dot(r, r))
        plane1 = np.cross(q, r)
        plane2 = np.cross(r, s)
        dihedral = np.arccos(np.dot(
            plane1, plane2) / (np.sqrt(np.dot(plane1, plane1)) * np.sqrt(np.dot(plane2, plane2))))
        if np.dot(np.cross(plane1, plane2), r_u) < 0 :
            dihedral = -dihedral

        return np.degrees(dihedral)

    # Поиск близжайших атомов в двух спсиках
    def contactpair(atoms1, atoms2):
        length = 1000
        count1 = 0
        for atom1, xyz1, atomnumber1 in atoms1:
            for atom2, xyz2, atomnumber2 in atoms2:
                currentlength = (m.sqrt(pow((float(xyz1[0]) - float(xyz2[0])), 2) +
                                        pow((float(xyz1[1]) - float(xyz2[1])), 2) +
                                        pow((float(xyz1[2]) - float(xyz2[2])), 2)))
                if currentlength < length:
                    atomout1 = atom1
                    atomout2 = atom2
                    xyzout1 = xyz1
                    xyzout2 = xyz2
                    length = currentlength
            count1 = count1 + 1

        out = [atomout1, xyzout1, atomout2, xyzout2, length]
        return out

    def center_cartesian(cartesian):
        """Find the center of mass and move it to the origin"""
        masses = {'X': 0, 'Ac': 227.028, 'Al': 26.981539, 'Am': 243, 'Sb': 121.757, 'Ar': 39.948,
                  'As': 74.92159, 'At': 210, 'Ba': 137.327, 'Bk': 247, 'Be': 9.012182, 'Bi': 208.98037,
                  'Bh': 262, 'B': 10.811, 'Br': 79.904, 'Cd': 112.411, 'Ca': 40.078, 'Cf': 251, 'C': 12.011,
                  'Ce': 140.115, 'Cs': 132.90543, 'Cl': 35.4527, 'Cr': 51.9961, 'Co': 58.9332, 'Cu': 63.546,
                  'Cm': 247, 'Db': 262, 'Dy': 162.5, 'Es': 252, 'Er': 167.26, 'Eu': 151.965, 'Fm': 257,
                  'F': 18.9984032, 'Fr': 223, 'Gd': 157.25, 'Ga': 69.723, 'Ge': 72.61, 'Au': 196.96654,
                  'Hf': 178.49, 'Hs': 265, 'He': 4.002602, 'Ho': 164.93032, 'H': 1.00794, 'In': 114.82,
                  'I': 126.90447, 'Ir': 192.22, 'Fe': 55.847, 'Kr': 83.8, 'La': 138.9055, 'Lr': 262,
                  'Pb': 207.2, 'Li': 6.941, 'Lu': 174.967, 'Mg': 24.305, 'Mn': 54.93805,
                  'Mt': 266, 'Md': 258, 'Hg': 200.59, 'Mo': 95.94, 'Nd': 144.24, 'Ne': 20.1797, 'Np': 237.048,
                  'Ni': 58.6934, 'Nb': 92.90638, 'N': 14.00674, 'No': 259, 'Os': 190.2, 'O': 15.9994,
                  'Pd': 106.42, 'P': 30.973762, 'Pt': 195.08, 'Pu': 244, 'Po': 209, 'K': 39.0983,
                  'Pr': 140.90765, 'Pm': 145, 'Pa': 231.0359, 'Ra': 226.025, 'Rn': 222, 'Re': 186.207,
                  'Rh': 102.9055, 'Rb': 85.4678, 'Ru': 101.07, 'Rf': 261, 'Sm': 150.36, 'Sc': 44.95591,
                  'Sg': 263, 'Se': 78.96, 'Si': 28.0855, 'Ag': 107.8682, 'Na': 22.989768, 'Sr': 87.62,
                  'S': 32.066, 'Ta': 180.9479, 'Tc': 98, 'Te': 127.6, 'Tb': 158.92534, 'Tl': 204.3833,
                  'Th': 232.0381, 'Tm': 168.93421, 'Sn': 118.71, 'Ti': 47.88, 'W': 183.85, 'U': 238.0289,
                  'V': 50.9415, 'Xe': 131.29, 'Yb': 173.04, 'Y': 88.90585, 'Zn': 65.39, 'Zr': 91.224}

        total_mass = 0.0
        center_of_mass = np.array([0.0, 0.0, 0.0])
        for atom, xyz, count in cartesian:
            total_mass += masses[atom]
            center_of_mass += xyz * masses[atom]
        center_of_mass = center_of_mass / total_mass

        out = ['X', center_of_mass, count]
        return out

    num2nam = {89 : 'Ac', 13 : 'Al', 95 : 'Am', 51 : 'Sb', 18 : 'Ar', 33 : 'As', 85 : 'At', 56 : 'Ba', 97 : 'Bk',
                       4 : 'Be',
                       83 : 'Bi', 107 : 'Bh', 5 : 'B', 35 : 'Br', 48 : 'Cd', 20 : 'Ca', 98 : 'Cf', 6 : 'C', 58 : 'Ce',
                       55 : 'Cs',
                       17 : 'Cl', 24 : 'Cr', 27 : 'Co', 29 : 'Cu', 96 : 'Cm', 105 : 'Db', 66 : 'Dy', 99 : 'Es', 68 : 'Er',
                       63 : 'Eu',
                       100 : 'Fm', 9 : 'F', 87 : 'Fr', 64 : 'Gd', 31 : 'Ga', 32 : 'Ge', 79 : 'Au', 72 : 'Hf', 108 : 'Hs',
                       2 : 'He',
                       67 : 'Ho', 1 : 'H', 49 : 'In', 53 : 'I', 77 : 'Ir', 26 : 'Fe', 36 : 'Kr', 57 : 'La', 103 : 'Lr',
                       82 : 'Pb',
                       3 : 'Li', 71 : 'Lu', 12 : 'Mg', 25 : 'Mn', 109 : 'Mt', 101 : 'Md', 80 : 'Hg', 42 : 'Mo', 60 : 'Nd',
                       10 : 'Ne',
                       93 : 'Np', 28 : 'Ni', 41 : 'Nb', 7 : 'N', 102 : 'No', 76 : 'Os', 8 : 'O', 46 : 'Pd', 15 : 'P',
                       78 : 'Pt',
                       94 : 'Pu', 84 : 'Po', 19 : 'K', 59 : 'Pr', 61 : 'Pm', 91 : 'Pa', 88 : 'Ra', 86 : 'Rn', 75 : 'Re',
                       45 : 'Rh',
                       37 : 'Rb', 44 : 'Ru', 104 : 'Rf', 62 : 'Sm', 21 : 'Sc', 106 : 'Sg', 34 : 'Se', 14 : 'Si', 47 : 'Ag',
                       11 : 'Na',
                       38 : 'Sr', 16 : 'S', 73 : 'Ta', 43 : 'Tc', 52 : 'Te', 65 : 'Tb', 81 : 'Tl', 90 : 'Th', 69 : 'Tm',
                       50 : 'Sn',
                       22 : 'Ti', 74 : 'W', 92 : 'U', 23 : 'V', 54 : 'Xe', 70 : 'Yb', 39 : 'Y', 30 : 'Zn', 40 : 'Zr'}
    atoms_order = []
    usingatoms = []
    order = []
    zmatrix = []
    zmatrix2 = []
    if atomsinfirst == 'Dimer' :
        cartesiandimer = read_cartesian(xyz1)
        atomsinsystem = len(cartesiandimer)//2
        cartesian = cartesiandimer[:atomsinsystem]
        cartesiansechalf = cartesiandimer[atomsinsystem:]

        center1 = closestatom_cartesian(cartesian, center_cartesian(cartesian), 0)
        center2 = closestatom_cartesian(cartesiansechalf, center_cartesian(cartesiansechalf), 0)

        if len(cartesiandimer) > atomsinsystem:
            contactdimer = contactpair(cartesian, cartesiansechalf)
        cartesiandimer2 = read_cartesian(xyz2)
        cartesian2 = cartesiandimer2[:atomsinsystem]
        cartesian2sechalf = cartesiandimer2[atomsinsystem:]
    else:
        cartesian = read_cartesian(xyz1)
        atomsinsystem = len(cartesian)
        cartesian2 = read_cartesian(xyz2)

    count = 1

    cartesianbackup = copy.deepcopy(cartesian)
    firstatom = cartesian[0]

    ''' first '''
    first = cartesian[0]
    first.append(count-1)
    first.append(count)
    first.append(count)
    first.append(count)
    workspace1 = first
    atoms_order.append(first[6])
    order.append([[], [], []])
    zmatrix.append([first[0], [[], [], []]])
    zmatrix2.append([cartesian2[atoms_order[0]-1][0], [[], [], []]])
    usingatoms.append(first)
    del cartesian[0]
    count = count + 1

    ''' second '''
    second = closestatom_cartesian(cartesian, first, count)
    workspace2 = second
    usingatoms.append(second)
    atoms_order.append(second[6])
    order.append([first[4], [], []])
    del cartesian[second[3]]
    zmatrix.append([second[0], [[first[4], second[2]], [], []]])
    zmatrix2.append([cartesian2[atoms_order[1] - 1][0], [[order[1][0], lengthcalc(cartesian2[atoms_order[0]-1], cartesian2[atoms_order[1] - 1])], [], []]])
    count = count + 1

    ''' third '''
    third = closestatom_cartesian(cartesian, first, count)
    workspace3 = third
    zmatrix.append([third[0], [[first[4], third[2]], [second[4], angle(third, first, second)], []]])
    usingatoms.append(third)
    atoms_order.append(third[6])
    order.append([first[4], second[4], []])
    zmatrix2.append([cartesian2[atoms_order[2] - 1][0],
                     [[order[2][0], lengthcalc(cartesian2[atoms_order[0] - 1], cartesian2[atoms_order[2] - 1])],
                      [order[2][1], angle(cartesian2[atoms_order[2] - 1], cartesian2[atoms_order[0] - 1], cartesian2[atoms_order[1] - 1])],
                      []]])
    del cartesian[third[3]]
    count = count + 1

    ''' fourth '''
    fourth = closestatom_cartesian(cartesian, first, count)
    workspace4 = fourth
    zmatrix.append([fourth[0], [[first[4], fourth[2]], [second[4], angle(fourth, first, second)], [third[4], dihedral(fourth, first, second, third)]]])
    usingatoms.append(fourth)
    atoms_order.append(fourth[6])
    order.append([first[4], second[4], third[4]])
    zmatrix2.append([cartesian2[atoms_order[3] - 1][0],
                     [[order[3][0], lengthcalc(cartesian2[atoms_order[0] - 1], cartesian2[atoms_order[3] - 1])],
                      [order[3][1], angle(cartesian2[atoms_order[3] - 1], cartesian2[atoms_order[0] - 1], cartesian2[atoms_order[1] - 1])],
                      [order[3][2], dihedral(cartesian2[atoms_order[3] - 1], cartesian2[atoms_order[0] - 1], cartesian2[atoms_order[1] - 1],  cartesian2[atoms_order[2] - 1])]]])

    del cartesian[fourth[3]]
    count = count + 1

    while cartesian:
        atom1 = closestatom_cartesian(cartesian, workspace1, count)
        atom2 = closestatom_cartesian(cartesian, workspace2, count)
        if atom1[2] :  # если у первого есть связь короче, берем ее
            atoms_order.append(atom1[6])
            # 1) workspace1: find closest: it's length
            workspace1 = closestatom_using(usingatoms, atom1)
            del usingatoms[workspace1[5]]
            length = workspace1[2]

            # 2) workspace2: find closest to workspace1: it's angle (workspace2:workspace1:atom2)
            workspace2 = closestatom_using(usingatoms, workspace1)
            del usingatoms[workspace2[5]]
            ang = angle(atom1, workspace1, workspace2)

            # 3) workspace2: find closest to workspace2: it's dihedral angle (workspace3:workspace2:workspace1:atom2)
            workspace3 = closestatom_using(usingatoms, workspace2)
            del usingatoms[workspace3[5]]
            dih = dihedral(atom1, workspace1, workspace2, workspace3)

            usingatoms.append(workspace1)
            usingatoms.append(workspace2)
            usingatoms.append(workspace3)
            usingatoms.append(atom1)
            del cartesian[atom1[3]]
            zmatrix.append([atom1[0], [[workspace1[4], length], [workspace2[4], ang], [workspace3[4], dih]]])
            order.append([workspace1[4], workspace2[4], workspace3[4]])
            zmatrix2.append([cartesian2[atoms_order[count-1] - 1][0],
                             [[order[count-1][0], lengthcalc(cartesian2[workspace1[6] - 1],
                                                         cartesian2[atom1[6] - 1])],
                              [order[count-1][1], angle(cartesian2[atom1[6] - 1],
                                                        cartesian2[workspace1[6] - 1],
                                                        cartesian2[workspace2[6] - 1])],
                              [order[count-1][2], dihedral(cartesian2[atom1[6] - 1],
                                                           cartesian2[workspace1[6] - 1],
                                                           cartesian2[workspace2[6] - 1],
                                                           cartesian2[workspace3[6] - 1])]]])

        else:
            atoms_order.append(atom2[6])
            # 1) workspace1: find closest: it's length
            workspace1 = closestatom_using(usingatoms, atom2)
            del usingatoms[workspace1[5]]
            length = workspace1[2]

            # 2) workspace2: find closest to workspace1: it's angle (workspace2:workspace1:atom2)
            workspace2 = closestatom_using(usingatoms, workspace1)
            del usingatoms[workspace2[5]]
            ang = angle(atom2, workspace1, workspace2)

            # 3) workspace2: find closest to workspace2: it's dihedral angle (workspace3:workspace2:workspace1:atom2)
            workspace3 = closestatom_using(usingatoms, workspace2)
            del usingatoms[workspace3[5]]
            dih = dihedral(atom2, workspace1, workspace2, workspace3)

            usingatoms.append(workspace1)
            usingatoms.append(workspace2)
            usingatoms.append(workspace3)
            usingatoms.append(atom2)
            del cartesian[atom2[3]]
            zmatrix.append([atom2[0], [[workspace1[4], length], [workspace2[4], ang], [workspace3[4], dih]]])
            order.append([workspace1[4], workspace2[4], workspace3[4]])
            zmatrix2.append([cartesian2[atoms_order[count - 1] - 1][0],
                             [[order[count - 1][0], lengthcalc(cartesian2[workspace1[6] - 1],
                                                               cartesian2[atom2[6] - 1])],
                              [order[count - 1][1], angle(cartesian2[atom2[6] - 1],
                                                          cartesian2[workspace1[6] - 1],
                                                          cartesian2[workspace2[6] - 1])],
                              [order[count - 1][2], dihedral(cartesian2[atom2[6] - 1],
                                                             cartesian2[workspace1[6] - 1],
                                                             cartesian2[workspace2[6] - 1],
                                                             cartesian2[workspace3[6] - 1])]]])

        count = count + 1

    fixlen = len(atoms_order)

    # make link from mol1 to mol2 between then centers


    if atomsinfirst == 'Dimer':
        #first atom in second molecule
        for atom in usingatoms:
            if center2[1].all == atom[1].all:
                workspace1 = atom


        first = closestatom_cartesian(cartesiansechalf, cartesiansechalf[0], count)
        workspace1 = closestatom_using(usingatoms, usingatoms[0])  # первый из второй молекулы
        length = lengthcalc(first, workspace1)
        del usingatoms[workspace1[5]]
        workspace2 = closestatom_using(usingatoms, workspace1)
        ang = angle(first, workspace1, workspace2)
        del usingatoms[workspace2[5]]
        workspace3 = closestatom_using(usingatoms, workspace2)
        dih = dihedral(first, workspace1, workspace2, workspace3)
        del usingatoms[workspace3[5]]
        order.append([workspace1[4], workspace2[4], workspace3[4]])
        zmatrix.append([first[0], [[workspace1[4], length], [workspace2[4], ang], [workspace3[4], dih]]])
        zmatrix2.append([cartesian2sechalf[first[6] - fixlen - 1][0],
                         [[order[count - 1][0], lengthcalc(cartesian2[workspace1[6] - 1 - fixlen],
                                                           cartesian2sechalf[first[6] - 1 - fixlen])],
                          [order[count - 1][1], angle(cartesian2sechalf[first[6] - 1 - fixlen],
                                                      cartesian2[workspace1[6] - 1 - fixlen],
                                                      cartesian2[workspace2[6] - 1 - fixlen])],
                          [order[count - 1][2], dihedral(cartesian2sechalf[first[6] - 1 - fixlen],
                                                         cartesian2[workspace1[6] - 1 - fixlen],
                                                         cartesian2[workspace2[6] - 1 - fixlen],
                                                         cartesian2[workspace3[6] - 1 - fixlen])]]])

        del cartesiansechalf[first[3]] # удалить по индексу начального списка
        usingatoms.clear()
        usingatoms.append(workspace1)
        usingatoms.append(workspace2)
        usingatoms.append(workspace3)
        usingatoms.append(first)
        atoms_order.append(first[6])
        count = count + 1

        second = closestatom_cartesian(cartesiansechalf, first, count)
        workspace1 = closestatom_using(usingatoms, second)
        length = workspace1[2]
        del usingatoms[workspace1[5]]
        workspace2 = closestatom_using(usingatoms, workspace1)
        ang = angle(second, workspace1, workspace2)
        del usingatoms[workspace2[5]]
        workspace3 = closestatom_using(usingatoms, workspace2)
        dih = dihedral(second, workspace1, workspace2, workspace3)
        del usingatoms[workspace3[5]]
        order.append([workspace1[4], workspace2[4], workspace3[4]])

        zmatrix.append([second[0], [[workspace1[4], length], [workspace2[4], ang], [workspace3[4], dih]]])
        zmatrix2.append([cartesian2sechalf[first[6] - fixlen - 1][0],
                         [[order[count - 1][0], lengthcalc(cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                           cartesian2sechalf[second[6] - 1 - fixlen])],
                          [order[count - 1][1], angle(cartesian2sechalf[second[6] - 1 - fixlen],
                                                      cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                      cartesian2[workspace2[6] - 1 - fixlen])],
                          [order[count - 1][2], dihedral(cartesian2sechalf[second[6] - 1 - fixlen],
                                                         cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                         cartesian2[workspace2[6] - 1 - fixlen],
                                                         cartesian2[workspace3[6] - 1 - fixlen])]]])

        del cartesiansechalf[second[3]]
        usingatoms.append(workspace1)
        usingatoms.append(workspace2)
        usingatoms.append(workspace3)
        usingatoms.append(second)
        atoms_order.append(second[6])
        count = count + 1

        third = closestatom_cartesian(cartesiansechalf, second, count)
        workspace1 = closestatom_using(usingatoms, third)
        length = workspace1[2]
        del usingatoms[workspace1[5]]
        workspace2 = closestatom_using(usingatoms, workspace1)
        ang = angle(third, workspace1, workspace2)
        del usingatoms[workspace2[5]]
        workspace3 = closestatom_using(usingatoms, workspace2)
        dih = dihedral(third, workspace1, workspace2, workspace3)
        del usingatoms[workspace3[5]]
        order.append([workspace1[4], workspace2[4], workspace3[4]])

        zmatrix.append([third[0], [[workspace1[4], length], [workspace2[4], ang], [workspace3[4], dih]]])
        zmatrix2.append([cartesian2sechalf[third[6] - fixlen - 1][0],
                         [[order[count - 1][0], lengthcalc(cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                           cartesian2sechalf[third[6] - 1 - fixlen])],
                          [order[count - 1][1], angle(cartesian2sechalf[third[6] - 1 - fixlen],
                                                      cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                      cartesian2sechalf[workspace2[6] - 1 - fixlen])],
                          [order[count - 1][2], dihedral(cartesian2sechalf[third[6] - 1 - fixlen],
                                                         cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                         cartesian2sechalf[workspace2[6] - 1 - fixlen],
                                                         cartesian2[workspace3[6] - 1 - fixlen])]]])

        del cartesiansechalf[third[3]]
        usingatoms.append(workspace1)
        usingatoms.append(workspace2)
        usingatoms.append(workspace3)
        usingatoms.append(third)
        atoms_order.append(third[6])
        count = count + 1

        while cartesiansechalf:
            atom1 = closestatom_cartesian(cartesiansechalf, workspace1, count)
            atom2 = closestatom_cartesian(cartesiansechalf, workspace2, count)
            if atom1[2] :  # если у первого есть связь короче, берем ее
                atoms_order.append(atom1[6])
                # 1) workspace1: find closest: it's length
                workspace1 = closestatom_using(usingatoms, atom1)
                del usingatoms[workspace1[5]]
                length = workspace1[2]

                # 2) workspace2: find closest to workspace1: it's angle (workspace2:workspace1:atom2)
                workspace2 = closestatom_using(usingatoms, workspace1)
                del usingatoms[workspace2[5]]
                ang = angle(atom1, workspace1, workspace2)

                # 3) workspace2: find closest to workspace2: it's dihedral angle (workspace3:workspace2:workspace1:atom2)
                workspace3 = closestatom_using(usingatoms, workspace2)
                del usingatoms[workspace3[5]]
                dih = dihedral(atom1, workspace1, workspace2, workspace3)

                usingatoms.append(workspace1)
                usingatoms.append(workspace2)
                usingatoms.append(workspace3)
                usingatoms.append(atom1)
                del cartesiansechalf[atom1[3]]
                zmatrix.append([atom1[0], [[workspace1[4], length], [workspace2[4], ang], [workspace3[4], dih]]])
                order.append([workspace1[4], workspace2[4], workspace3[4]])

                zmatrix2.append([cartesian2sechalf[atom1[6]-fixlen-1][0],
                                 [[order[count-1][0], lengthcalc(cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                             cartesian2sechalf[atom1[6] - 1 - fixlen])],
                                  [order[count-1][1], angle(cartesian2sechalf[atom1[6] - 1 - fixlen],
                                                            cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                            cartesian2sechalf[workspace2[6] - 1 - fixlen])],
                                  [order[count-1][2], dihedral(cartesian2sechalf[atom1[6] - 1 - fixlen],
                                                               cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                               cartesian2sechalf[workspace2[6] - 1 - fixlen],
                                                               cartesian2sechalf[workspace3[6] - 1 - fixlen])]]])


            else:
                atoms_order.append(atom2[6])
                # 1) workspace1: find closest: it's length
                workspace1 = closestatom_using(usingatoms, atom2)
                del usingatoms[workspace1[5]]
                length = workspace1[2]

                # 2) workspace2: find closest to workspace1: it's angle (workspace2:workspace1:atom2)
                workspace2 = closestatom_using(usingatoms, workspace1)
                del usingatoms[workspace2[5]]
                ang = angle(atom2, workspace1, workspace2)

                # 3) workspace2: find closest to workspace2: it's dihedral angle (workspace3:workspace2:workspace1:atom2)
                workspace3 = closestatom_using(usingatoms, workspace2)
                del usingatoms[workspace3[5]]
                dih = dihedral(atom2, workspace1, workspace2, workspace3)

                usingatoms.append(workspace1)
                usingatoms.append(workspace2)
                usingatoms.append(workspace3)
                usingatoms.append(atom2)
                del cartesiansechalf[atom2[3]]
                zmatrix.append([atom2[0], [[workspace1[4], length], [workspace2[4], ang], [workspace3[4], dih]]])
                order.append([workspace1[4], workspace2[4], workspace3[4]])

                zmatrix2.append([cartesian2sechalf[atom2[6]-fixlen-1][0],
                                 [[order[count - 1][0], lengthcalc(cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                                   cartesian2sechalf[atom2[6] - 1 - fixlen])],
                                  [order[count - 1][1], angle(cartesian2sechalf[atom2[6] - 1 - fixlen],
                                                              cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                              cartesian2sechalf[workspace2[6] - 1 - fixlen])],
                                  [order[count - 1][2], dihedral(cartesian2sechalf[atom2[6] - 1 - fixlen],
                                                                 cartesian2sechalf[workspace1[6] - 1 - fixlen],
                                                                 cartesian2sechalf[workspace2[6] - 1 - fixlen],
                                                                 cartesian2sechalf[workspace3[6] - 1 - fixlen])]]])


            count = count + 1


    def str_zmatrix(zmatrix) :
        out = 'text\nC1\n'
        izmatout = ' $ZMAT   IZMAT(1)=\n'
        out += zmatrix[0][0] + '\n'
        out += '{:<4}{:<5}{:<15.10f}\n'.format(zmatrix[1][0], zmatrix[1][1][0][0], zmatrix[1][1][0][1])
        izmatout += '         {:>4},{:>4},{:>4},\n'.format('1', '2', zmatrix[1][1][0][0])
        out += '{:<4}{:<5}{:<15.10f}{:<5}{:<15.10f}\n'.format(zmatrix[2][0], zmatrix[2][1][0][0], zmatrix[2][1][0][1], zmatrix[2][1][1][0], zmatrix[2][1][1][1])
        izmatout += '         {:>4},{:>4},{:>4},\n'.format('1', '3', zmatrix[2][1][0][0], )
        izmatout += '         {:>4},{:>4},{:>4},{:>4},\n'.format('2', '3', zmatrix[2][1][0][0], zmatrix[2][1][1][0])
        out += '{:<4}{:<5}{:<15.10f}{:<5}{:<15.10f}{:<5}{:<15.10f}\n'\
            .format(zmatrix[3][0], zmatrix[3][1][0][0], zmatrix[3][1][0][1], zmatrix[3][1][1][0], zmatrix[3][1][1][1], zmatrix[3][1][2][0], zmatrix[3][1][2][1])
        izmatout += '         {:>4},{:>4},{:>4},\n'.format('1', '4', zmatrix[3][1][0][0], )
        izmatout += '         {:>4},{:>4},{:>4},{:>4},\n'.format('2', '4', zmatrix[3][1][0][0], zmatrix[3][1][1][0])
        izmatout += '         {:>4},{:>4},{:>4},{:>4},{:>4},\n'.format('3', '4', zmatrix[3][1][0][0], zmatrix[3][1][1][0], zmatrix[3][1][2][0])
        for i in range(4, len(zmatrix)):
            out += '{:<4}{:<5}{:<15.10f}{:<5}{:<15.10f}{:<5}{:<15.10f}\n'\
            .format(zmatrix[i][0], zmatrix[i][1][0][0], zmatrix[i][1][0][1], zmatrix[i][1][1][0], zmatrix[i][1][1][1], zmatrix[i][1][2][0], zmatrix[i][1][2][1])
            izmatout += '         {:>4},{:>4},{:>4},\n'.format('1', i+1, zmatrix[i][1][0][0], )
            izmatout += '         {:>4},{:>4},{:>4},{:>4},\n'.format('2', i+1, zmatrix[i][1][0][0], zmatrix[i][1][1][0])
            izmatout += '         {:>4},{:>4},{:>4},{:>4},{:>4},\n'.format('3', i+1, zmatrix[i][1][0][0],
                                                                           zmatrix[i][1][1][0], zmatrix[i][1][2][0])

        izmatout += ' $END\n'
        return [out, izmatout]

    with open(outfile1, 'w') as f :
        f.write(str_zmatrix(zmatrix)[0])

    with open(outfile2, 'w') as g :
        g.write(str_zmatrix(zmatrix2)[0])

    with open('izmat.tmp', 'w') as izmat :
        izmat.write(str_zmatrix(zmatrix)[1])


def file_generator(settings_file, data_file_left, data_file_right, path, step, outfile_mask, atomsinfirstmolecule):

    #settings_file = 'C:/Users/ilya_/PycharmProjects/chemapp/venv/settings.inp'
    #data_file_left = r'C:\Users\ilya_\PycharmProjects\chemapp\venv\tpbi1.xyz'
    #data_file_right = r'C:\Users\ilya_\PycharmProjects\chemapp\venv\tpbi2.xyz'
    #path = 'C:/Users/ilya_/PycharmProjects/chemapp/venv/inpfiles'
    if outfile_mask == '':
        outfile_mask = 'default'

    if step == 0 :
        step = 0.1

    totaldict = dict()
    a = Converter()

    zmatrixgenerator(data_file_left, data_file_right, 'zmatrix1.dat', 'zmatrix2.dat', atomsinfirstmolecule)

    def formatzmt(readfile, writefile):
        '''Извлечение данных из файла с z матрицей'''
        with open(readfile, 'r') as f:
            f.readline()
            f.readline()

            name1 = f.readline()

            line2 = f.readline()
            atom = line2.split()
            name2 = atom[0]
            num1 = atom[1]
            length2 = atom[2]

            line3 = f.readline()
            atom = line3.split()
            name3 = atom[0]
            num2 = atom[1]
            length3 = atom[2]
            num3 = atom[3]
            angle3 = atom[4]

            j = 0
            name = []
            nums1 = []
            length = []
            nums2 = []
            angle = []
            nums3 = []
            dihedral = []
            signs = []
            for line in f.readlines():
                atom = line.split()
                name.append(atom[0])
                nums1.append(atom[1])
                length.append(atom[2])
                nums2.append(atom[3])
                angle.append(atom[4])
                nums3.append(atom[5])
                dihedral.append(atom[6])
                j = j + 1

            for i in range(len(dihedral)):
                if (float(dihedral[i - 3]) > -270 and float(dihedral[i - 3]) < -90):
                    signs.append('-')
                    dihedral[i - 3] = str((float(dihedral[i - 3])) + 360)
                else:
                    signs.append('')

        '''Формирование нового файла в формате GAMESS'''
        zmt = open(writefile, 'w+')
        zmt.write(' $DATA\nZMATRIX\nC1\n')

        i = 0
        zmt.write('%s' % name1)
        i = i + 1

        key = 'R' + str(i) + '_' + str(i + 1)
        zmt.write('{:<4}{:<5}{:<15}\n'.format(name2, num1, key))
        totaldict[key] = length2
        i = i + 1

        key1 = 'R' + str(i) + '_' + str(i + 1)
        key2 = 'A' + str(i + 1) + '_' + str(i) + '_' + str(i - 1)
        zmt.write('{:<4}{:<5}{:<15}{:<5}{:<15}\n'.format(name3, num2, key1, num3, key2))
        totaldict[key1] = length3
        totaldict[key2] = angle3
        i = i + 1

        for line in range(len(name)):
            key1 = 'R' + str(i) + '_' + str(i + 1)
            key2 = 'A' + str(i + 1) + '_' + str(i) + '_' + str(i - 1)
            key3 = 'D' + str(i + 1) + '_' + str(i) + '_' + str(i - 1) + '_' + str(i - 2)
            zmt.write('{:<4}{:<5}{:<15}{:<5}{:<15}{:<5}{:<15}\n'
                      .format(name[i - 3], nums1[i-3], key1, nums2[i-3], key2, nums3[i-3], key3))
            totaldict[key1] = length[i - 3]
            totaldict[key2] = angle[i - 3]
            totaldict[key3] = dihedral[i - 3]
            i = i + 1
        zmt.write('\n')

        for key, value in totaldict.items():
            zmt.write('{:<12}    =     {:<.12}\n'.format(str(key), str(value)))
        zmt.write(' $END\n')

        zmt.close()

    formatzmt('zmatrix1.dat', 'zmatrix1.tmp')
    formatzmt('zmatrix2.dat', 'zmatrix2.tmp')

    # get values
    def reformatzmt(inpfile) :
        inp = open(inpfile, 'r+')
        out = open(inpfile[:-4] + 'out.txt', 'w+')
        out.write(inp.readline())
        out.write(inp.readline())
        out.write(inp.readline())
        out.write(inp.readline())
        out.write(inp.readline())
        out.write(inp.readline())
        out.write(inp.readline())

        dihedralchanges = []
        for line in inp :
            if line == '\n' :
                break
            else :
                atom, num1, len, num2, angle, num3, dihedral = line.split()
                if re.match('-D', dihedral) :
                    dihedral = re.sub('-D', 'D', dihedral)
                    dihedralchanges.append(dihedral)

                line = ('{:<4}{:<5}{:<15}{:<5}{:<15}{:<5}{:<15}\n'.format(atom, num1, len, num2, angle, num3, dihedral))
                out.write(line)

        out.write('\n')
        codes = []
        values = []
        for line in inp :
            if line == ' $END\n' :
                break
            else :
                code, value = line.split('=')
                code = code.strip()
                codes.append(code)
                if code in dihedralchanges :
                    value = value.strip()
                    value = '-' + value
                    values.append(float(value))
                else :
                    value = value.strip()
                    values.append(float(value))

            line = '{:<12}    =     {:<.15}\n'.format(str(code), str(value))
            out.write(line)

        out.write(' $END\n')
        for line in inp :
            out.write(line)

        inp.close()
        out.close()

        return values

    geometryLeft = reformatzmt('zmatrix1.tmp')
    geometryRight = reformatzmt('zmatrix2.tmp')

    def geometries(geometryLeft, geometryRight, step):
        # функция, создающая таблицу с промежуточными геометриями
        #f = open(FILE_A, 'r')
        #g = open(FILE_B, 'r')

        i = 0
        for item in range(len(geometryLeft)):
            if (i>3 and (i - 2) % 3 == 0):
                if -270.0 < float(geometryLeft[item]) < -90.0:
                    geometryLeft[item] = float(geometryLeft[item]) + 360.0
                if -270.0 < float(geometryRight[item]) < -90.0:
                    geometryRight[item] = float(geometryRight[item]) + 360.0
            i = i + 1

        amoutofpoints = int(1 / float(step) - 1)

        points = [0.0]
        for i in range(amoutofpoints + 1):
            points.append("{0:.2f}".format((float(step) * (i + 1))))


        intgeom = []
        for i in range(len(points) + 2):
            intgeom.append([])
            for j in range(len(geometryLeft)):
                intgeom[i].append(
                    float("{0:.6f}".format((i * float(step)) * float(geometryLeft[j]) + (1 - i * float(step)) * float(geometryRight[j]))))

        OUT_FILE = 'tempgeom.txt'

        t = open(OUT_FILE, 'w')
        for i in range(len(points)):
            t.write("%10s\t" % points[i])

        t.write("\n")
        for i in range(len(geometryLeft)):
            for j in range(len(points)):
                t.write("%10s\t" % intgeom[j][i])
            t.write("\n")

        t.close()
        return intgeom

    geom = geometries(geometryLeft, geometryRight, step)

    # начинаем запись новых файлов
    f = open('zmatrix1.tmp', 'r')
    g = open(settings_file, 'r')

    amoutofpoints = int(1 / float(step) + 1)

    path = path + '/' + outfile_mask
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    def gamzmt2zmt(zmtfile, outzmtfile) :
        inpfile = open(zmtfile, 'r')
        outfile = open(outzmtfile, 'w+')
        flag1 = 0
        codes = []
        values = []
        for line in inpfile :
            if line == '\n' :
                flag1 = 1

            elif flag1 == 1 :
                if line == ' $END\n' :
                    flag1 = 2
                else :
                    code, value = line.split('=')
                    code = code.strip()
                    codes.append(code)
                    value = value.strip()
                    values.append(value)
            elif flag1 == 2 :
                break
            else :
                continue

        dictvalues = dict(zip(codes, values))
        inpfile.seek(0)
        flag1 = 0
        for line in inpfile :
            if line == ' $DATA\n' :
                inpfile.readline()
                inpfile.readline()
                outfile.write(inpfile.readline())
                line = inpfile.readline()
                atom, num1, length = line.split()
                length = dictvalues.get(length)
                outfile.write('{:<4}{:<5}{:<15.15s}\n'.format(atom, num1, length))
                line = inpfile.readline()
                atom, num1, length, num2, angle = line.split()
                length = dictvalues.get(length)
                angle = dictvalues.get(angle)
                outfile.write('{:<4}{:<5}{:<15.15s}{:<5}{:<15.15s}\n'.format(atom, num1, length, num2, angle))
                flag1 = 1
            elif flag1 == 1 :
                if line == '\n' :
                    flag1 = 2
                else :
                    atom, num1, length, num2, angle, num3, dihedral = line.split()
                    length = dictvalues.get(length)
                    angle = dictvalues.get(angle)
                    dihedral = re.sub(r'-D', 'D', dihedral)
                    dihedral = dictvalues.get(dihedral)
                    outfile.write(
                        '{:<4}{:<5}{:<15.15s}{:<5}{:<15.15s}{:<5}{:<15.15s}\n'.format(atom, num1, length, num2, angle,
                                                                                      num3, dihedral))

        outfile.close()
        inpfile.close()

    izmat = open('izmat.tmp', 'r')

    for k in range(amoutofpoints):
        if amoutofpoints <= 9:
            current_step = k * int(step * 10)
            if (current_step) < 9:
                outfile = path + '\%s' % outfile_mask + '_0%s0' % str(current_step) + 'zmt.inp'
            else:
                outfile = path + '\%s' % outfile_mask + '_%s0' % str(current_step) + 'zmt.inp'
        elif 9 < amoutofpoints <= 101:
            current_step = k * int(step * 100)
            if current_step <= 9:
                outfile = path + '\%s' % outfile_mask + '_00%s' % str(current_step) + 'zmt.inp'
            elif 9 < current_step <= 99:
                outfile = path + '\%s' % outfile_mask + '_0%s' % str(current_step) + 'zmt.inp'
            else:
                outfile = path + '\%s' % outfile_mask + '_%s' % str(current_step) + 'zmt.inp'

        t = open(outfile, 'w')

        f.seek(0)
        g.seek(0)

        # write settings
        for lines in g:
            t.write(lines)

        for line in f:
            if re.search('DATA', line):
                break

        t.write('\n $DATA\nTitle\n')

        # вставка группы DATA
        temp = 0
        for i in f:
            if temp == 0:
                temp += 1
            elif i.strip():
                #i = re.sub(r'-D', 'D', i)
                t.write(i)
                temp += 1
            else:
                t.write("\n")
                temp += 1
                break
        
        i = 0
        for line in f:
            if line == ' $END\n' :
                break
            else :
                code = line.split('=')[0]
                code = code.strip()
                line = '{:<12}    =     {:<.15}\n'.format(str(code), str(geom[k][i]))
                i = i + 1
                t.write(line)


        t.write(" $END\n")

        for line in f:
            t.write(line)

        t.close()

        # Смена формата z матрицы
        gamzmt2zmt(outfile, outfile[:-4] + 'zmt.tmp')

        a.run_zmatrix(input_file=outfile[:-4] + 'zmt.tmp', output_file=outfile[:-7]+'.xyz')
        os.remove(outfile[:-4] + 'zmt.tmp')

        # переделать inp файлы с декартовыми координатами
        zmt2xyz(outfile)
        t = open(outfile[:-7] + '.inp', 'a')
        izmat.seek(0)
        for line in izmat:
            t.write(line)

        t.close()

        os.remove(outfile[:-4] + '.inp')

    zmttraj(path)

    izmat.close()
    f.close()

    os.mkdir(path+'/xyz')

    for filename in glob.glob(os.path.join(path, '*.xyz')):
        shutil.move(filename, path+'/xyz')

    os.remove('izmat.tmp')
    os.remove('zmatrix1.tmp')
    os.remove('zmatrix2.tmp')
    os.remove('zmatrix1.dat')
    os.remove('zmatrix2.dat')
    os.remove('zmatrix1out.txt')
    os.remove('zmatrix2out.txt')
    os.remove('tempgeom.txt')


def method_changer(path, NEW_SET) :
    numberoffiles = 0
    filenames = []
    for filename in glob.glob(os.path.join(path, '*.inp')) :
        filenames.append(str(filename))
        numberoffiles += 1

    for num in filenames :

        f = open(num, 'r+')
        g = open(NEW_SET, 'r')
        t = open('tempfile.txt', 'w+')

        for line in g :
            t.write(line)

        for line in f :
            if re.search('DATA', line) :
                break
            else :
                continue

        t.write('\n $DATA\n')
        for line in f :
            t.write(line)

        f.close()
        os.remove(num)
        f = open(num, 'w')
        t.seek(0)

        for line in t :
            f.write(line)

        f.close()
        g.close()
        t.close()
        os.remove('tempfile.txt')


def vec_changer(path, VEC_FILE) :
    numberoffiles = 0
    filenames = []
    for filename in glob.glob(os.path.join(path, '*.inp')) :
        filenames.append(str(filename))
        numberoffiles += 1

    for num in filenames :
        f = open(num, 'r+')
        g = open(VEC_FILE, 'r')
        t = open('tempfile.txt', 'w+')

        for line in f :
            if re.search('VEC', line) :
                f.close()
                os.remove(num)
                break
            else :
                t.write(line)

        flag2 = 0
        for line in g :
            if flag2 == 1 :
                t.write(line)
            elif re.search('VEC', line) :
                flag2 = 1
                t.write(line)
            else :
                continue
        f = open(num, 'w')
        f.seek(0)
        t.seek(0)
        for line in t :
            f.write(line)

        f.close()
        g.close()
        t.close()
        os.remove('tempfile.txt')


def zmt2xyz(zmtinput):
    ''' translate zmtinput to xyzinput '''

    zmt = open(zmtinput, 'r')
    xyz = open(zmtinput[:-7] + '.xyz', 'r')
    inp = open(zmtinput[:-7] + '.inp', 'w')

    nam2num = {'Ac': 89, 'Al': 13, 'Am': 95, 'Sb': 51, 'Ar': 18, 'As': 33, 'At': 85, 'Ba': 56, 'Bk': 97,
               'Be': 4, 'Bi': 83, 'Bh': 107, 'B': 5, 'Br': 35, 'Cd': 48, 'Ca': 20, 'Cf': 98, 'C': 6,
               'Ce': 58, 'Cs': 55, 'Cl': 17, 'Cr': 24, 'Co': 27, 'Cu': 29, 'Cm': 96, 'Db': 105, 'Dy': 66,
               'Es': 99, 'Er': 68, 'Eu': 63, 'Fm': 100, 'F': 9, 'Fr': 87, 'Gd': 64, 'Ga': 31, 'Ge': 32,
               'Au': 79, 'Hf': 72, 'Hs': 108, 'He': 2, 'Ho': 67, 'H': 1, 'In': 49, 'I': 53, 'Ir': 77,
               'Fe': 26, 'Kr': 36, 'La': 57, 'Lr': 103, 'Pb': 82, 'Li': 3, 'Lu': 71, 'Mg': 12, 'Mn': 25,
               'Mt': 109, 'Md': 101, 'Hg': 80, 'Mo': 42, 'Nd': 60, 'Ne': 10, 'Np': 93, 'Ni': 28, 'Nb': 41,
               'N': 7, 'No': 102, 'Os': 76, 'O': 8, 'Pd': 46, 'P': 15, 'Pt': 78, 'Pu': 94, 'Po': 84,
               'K': 19, 'Pr': 59, 'Pm': 61, 'Pa': 91, 'Ra': 88, 'Rn': 86, 'Re': 75, 'Rh': 45, 'Rb': 37,
               'Ru': 44, 'Rf': 104, 'Sm': 62, 'Sc': 21, 'Sg': 106, 'Se': 34, 'Si': 14, 'Ag': 47, 'Na': 11,
               'Sr': 38, 'S': 16, 'Ta': 73, 'Tc': 43, 'Te': 52, 'Tb': 65, 'Tl': 81, 'Th': 90, 'Tm': 69,
               'Sn': 50, 'Ti': 22, 'W': 74, 'U': 92, 'V': 23, 'Xe': 54, 'Yb': 70, 'Y': 39, 'Zn': 30,
               'Zr': 40}
    # Вписать полные имена
    fullnames = {'Ac': 'ACTINIUM', 'Al': 'ALUMINIUM', 'Am': 'AMERICIUM', 'Sb': 'ANTIMONY', 'Ar': 'ARGON',
                 'As': 'ARSENIC', 'At': 'ASTATINE', 'Ba': 'BARIUM', 'Bk': 'BERKELIUM', 'Be': 'BERYLLIUM',
                 'Bi': 'BISMUTH', 'Bh': 'BOHRIUM', 'B': 'BORON', 'Br': 'BROMINE', 'Cd': 'CADMIUM', 'Ca': 'CALCIUM',
                 'Cf': 'CALIFORNIUM', 'C': 'CARBON', 'Ce': 'CERIUM', 'Cs': 'CESIUM', 'Cl': 'CHLORINE', 'Cr': 'CHROMIUM',
                 'Co': 'COBALT', 'Cu': 'COPPER', 'Cm': 'CURIUM', 'Db': 'DUBNIUM', 'Dy': 'DYSPROSIUM',
                 'Es': 'EINSTEINIUM', 'Er': 'ERBIUM', 'Eu': 'EUROPEUM', 'Fm': 'FERMIUM', 'F': 'FLUORINE',
                 'Fr': 'FRANCIUM', 'Gd': 'GADOLINIUM', 'Ga': 'GALLIUM', 'Ge': 'GERMANIUM', 'Au': 'GOLD',
                 'Hf': 'HAFNIUM', 'Hs': 'HASSIUM', 'He': 'HELIUM', 'Ho': 'HOLMIUM', 'H': 'HYDROGEN', 'In': 'INDIUM',
                 'I': 'IODINE', 'Ir': 'IRIDIUM', 'Fe': 'IRON', 'Kr': 'KRYPTON', 'La': 'LANTHANUM', 'Lr': 'LAWRENCIUM',
                 'Pb': 'LEAD', 'Li': 'LITHIUM', 'Lu': 'LUTETIUM', 'Mg': 'MAGNESIUM', 'Mn': 'MANGANESE',
                 'Mt': 'MEITNERIUM', 'Md': 'MENDELEVIUM', 'Hg': 'MERCURY', 'Mo': 'MOLYBDENUM', 'Nd': 'NEODYMIUM',
                 'Ne': 'NEON', 'Np': 'NEPTUNIUM', 'Ni': 'NICKEL', 'Nb': 'NIOBIUM', 'N': 'NITROGEN', 'No': 'NOBELIUM',
                 'Os': 'OSMIUM', 'O': 'OXYGEN', 'Pd': 'PALLADIUM', 'P': 'PHOSPHORUS', 'Pt': 'PLATINUM',
                 'Pu': 'PLUTONIUM', 'Po': 'POLONIUM', 'K': 'POTASSIUM', 'Pr': 'PRASEODYMIUM', 'Pm': 'PROMETHIUM',
                 'Pa': 'PROACTINIUM', 'Ra': 'RADIUM', 'Rn': 'RADON', 'Re': 'RHENIUM', 'Rh': 'RHODIUM', 'Rb': 'RUBIDIUM',
                 'Ru': 'RUTHENIUM', 'Rf': 'RUTHERFORDIUM', 'Sm': 'SAMARIUM', 'Sc': 'SCANDIUM', 'Sg': 'SEABORGIUM',
                 'Se': 'SELENIUM', 'Si': 'SILICON', 'Ag': 'SILVER', 'Na': 'SODIUM', 'Sr': 'STRONTIUM', 'S': 'SULFUR',
                 'Ta': 'TANTALUM', 'Tc': 'TECHNETIUM', 'Te': 'TELLURIUM', 'Tb': 'TERBIUM', 'Tl': 'THALLIUM',
                 'Th': 'THORIUM', 'Tm': 'THULIUM', 'Sn': 'TIN', 'Ti': 'TITANIUM', 'W': 'TUNGSTEN', 'U': 'URANIUM',
                 'V': 'VANADIUM', 'Xe': 'XENON', 'Yb': 'YTTERBIUM', 'Y': 'YTTRIUM', 'Zn': 'ZINC', 'Zr': 'ZIRCONIUM'}

    flag = 0
    for line in zmt:
        if re.search('DATA', line) or flag == 1 or flag == 2:
            flag += 1
            inp.write(line)
        elif flag == 0:
            inp.write(line)

    flag = 0
    for line in xyz:
        if flag == 2:
            inp.write('{:<15}{:<10}{:<20}{:<20}{:<20}\n'
                       .format(str(fullnames[line.split()[0]]), float(nam2num[line.split()[0]]), line.split()[1],
                               line.split()[2], line.split()[3]))
        else:
            flag += 1

    inp.write(' $END\n')
    xyz.close()
    inp.close()


def zmttraj(path) :
    numberoffiles = 0
    filenames = []
    for filename in glob.glob(os.path.join(path, '*.xyz')) :
        filenames.append(str(filename))
        numberoffiles += 1

    trj = r"{0}\\trajectory{1}.xyz".format(path, str(numberoffiles))
    traj = open(trj, "w")
    i = 0
    mode = 'xyz'
    for num in filenames :
        if mode == 'GAMESSForm':
            i += 1
            f = open(num, 'r')
            flag = 0
            traj.write('Point ' + str(i) + '\n')
            for line in f :
                if re.search('DATA', line) or flag == 1 or flag == 2 :
                    flag += 1
                elif flag == 3 :
                    if re.search('END', line) :
                        break
                    else :
                        tmpline = line.split()
                        traj.write('{:<5}{:<20}{:<20}{:<20}\n'
                                   .format(str(tmpline[1]), str(tmpline[2]), str(tmpline[3]), str(tmpline[4])))
                elif flag == 0 :
                    continue

            traj.write('\n\n')
            f.close()
        elif mode == 'xyz':
            i += 1
            f = open(num, 'r')
            traj.write('Point ' + str(i) + '\n')
            for line in f :
                traj.write(line)

            traj.write('\n\n')
            f.close()


    traj.close()


def qdptresult(path, state1inp=1, state2inp=2, linestyle1='-', linewidth1=5.0, linecolor1='blue', marker1='D',
               markersize1='7',  linestyle2='-', linewidth2=5.0, linecolor2='orange', marker2='D', markersize2='7',
               ScaleFontsize='large', axeswidth=2.0, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, title='Energy profile'):

    #path = 'D:/НИРС/molecules/oxd7/XMCQDPT/newoxd/pair4'
    ''' INTITAL PARAMETERS '''

    ''' marker1 = 'D'
    marker2 = 'D'

    markersize1 = '7'
    markersize2 = '7'

    linecolor1 = 'blue'
    linecolor2 = 'orange'

    linewidth1 = 5
    linewidth2 = 5

    linestyle1 = '-'
    linestyle2 = '-'

    axeswidth=3.0
    title = 'Energy profile'
    x_min = -0.05
    x_max = 1.05
    y_min = 0.0
    y_max = 0.5 '''

    state_one = state1inp
    state_two = state2inp

    ''' PROGRAMM '''

    step = 0
    gap = []
    state1 = []
    state2 = []

    # считывание всех файлов в каталоге
    numberoffiles = 0
    filenames = []
    for filename in glob.glob(os.path.join(path, '*.out')) :
        filenames.append(str(filename))
        numberoffiles += 1

    xcoord = []
    for i in range(numberoffiles) :
        xcoord.append((float(filenames[i][-7:-4])/100))

    # вырезание энергий из файла
    energies = []
    i = 0
    for num in filenames :
        i += 1
        f = open(num, 'r')
        flag = 0
        for line in f :
            if re.search(' XMC-QDPT2 ENERGIES ', line) or flag == 1 :
                flag += 1
            elif flag == 2 :
                break
            else :
                continue
        flag = 0
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader :
            if re.search('-----------------------', str(row)) :
                break
            else :
                energies.append(row[19])
                flag += 1
        f.close()

    # перевод в eV
    val, idx = max((val, idx) for (idx, val) in enumerate(energies))
    for i in range(len(energies)) :
        energies[i] = str((float(energies[i]) - float(val)) * 27.2114)

    # запись в текстовый файл

    OUT_FILENAME = "energies.txt"
    t = open(OUT_FILENAME, 'w+')
    t.write('Energies of all states in eV\n')

    for i in range(flag) :
        for k in range(numberoffiles) :
            t.write("%20s" % str(energies[i + k * flag]))
            t.write('\t')
        t.write('\n')

    minstate = 100
    for i in range(numberoffiles) :
        state1.append((float(energies[state_one - 1 + step])))
        state2.append((float(energies[state_two - 1 + step])))
        if state1[i] < minstate:
            minstate = state1[i]
            numstate = i

        ereorg = state2[numstate] - state1[numstate]
        gap.append((float(energies[state_two - 1 + step]) - float(energies[state_one - 1 + step])))
        step += flag

    mingap = min(gap)

    for i in range(len(gap)):
        if gap[i] == min(gap):
            gapindex = i

    DeltaG = abs(min(state1[:gapindex+1]) - min(state1[-gapindex-1:]))

    #t.write('\nEnergy gap between state%d' % state_one)
    #t.write(' and state%d in eV\n ' % state_two)

    outparameters = [ereorg, mingap, DeltaG, xcoord, energies]

    for i in range(numberoffiles) :
        t.write(str(gap[i]))
        t.write(';')
    t.write('\n')

    # Построение энергетического профиля

    fig = plt.figure(num=None, figsize=(10, 8), facecolor='white', edgecolor='white', frameon=True, clear=True)
    plt.title(title, fontsize='xx-large')
    ax = fig.gca()
    ax.set_xlabel('Reaction coordinate', fontsize='xx-large')
    ax.set_ylabel('Energy (eV)', fontsize='xx-large')
    ax.tick_params(labelsize=ScaleFontsize)
    plt.axis((x_min, x_max, y_min, y_max))
    plt.setp(ax.spines.values(), linewidth=axeswidth)

    line1 = ax.plot(xcoord, state1)
    plt.setp(line1, marker=marker1, markersize=markersize1, color=linecolor1, linewidth=linewidth1,
             linestyle=linestyle1, label='State 1')
    line2 = ax.plot(xcoord, state2)
    plt.setp(line2, marker=marker2, markersize=markersize2, color=linecolor2, linewidth=linewidth2,
             linestyle=linestyle2, label='State 2')

    ax.legend(loc='center right', fontsize='large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
    #plt.savefig('graph.png', dpi=300)

    return outparameters


def packdelimiter(inputfile, minatomsinmol, maxcontact):
    #inputfile = input("MolecualrDelimeter&DimerBuilder\nEnter input filename: ")

    # minimum atoms in molecule
    #minatomsinmol = input("Enter minimum number of atoms in molecule: ")

    # max dimer contact length
    #maxcontact = input("Enter maximum contact length in dimer: ")

    outmonomer = str(inputfile)[:-4] + '_monomers.xyz'
    outdimer = str(inputfile)[:-4] + '_dimers.xyz'

    def read_cartesian(file):
        nam2num = {'Ac': 89, 'Al': 13, 'Am': 95, 'Sb': 51, 'Ar': 18, 'As': 33, 'At': 85, 'Ba': 56, 'Bk': 97,
                   'Be': 4, 'Bi': 83, 'Bh': 107, 'B': 5, 'Br': 35, 'Cd': 48, 'Ca': 20, 'Cf': 98, 'C': 6,
                   'Ce': 58, 'Cs': 55, 'Cl': 17, 'Cr': 24, 'Co': 27, 'Cu': 29, 'Cm': 96, 'Db': 105, 'Dy': 66,
                   'Es': 99, 'Er': 68, 'Eu': 63, 'Fm': 100, 'F': 9, 'Fr': 87, 'Gd': 64, 'Ga': 31, 'Ge': 32,
                   'Au': 79, 'Hf': 72, 'Hs': 108, 'He': 2, 'Ho': 67, 'H': 1, 'In': 49, 'I': 53, 'Ir': 77,
                   'Fe': 26, 'Kr': 36, 'La': 57, 'Lr': 103, 'Pb': 82, 'Li': 3, 'Lu': 71, 'Mg': 12, 'Mn': 25,
                   'Mt': 109, 'Md': 101, 'Hg': 80, 'Mo': 42, 'Nd': 60, 'Ne': 10, 'Np': 93, 'Ni': 28, 'Nb': 41,
                   'N': 7, 'No': 102, 'Os': 76, 'O': 8, 'Pd': 46, 'P': 15, 'Pt': 78, 'Pu': 94, 'Po': 84,
                   'K': 19, 'Pr': 59, 'Pm': 61, 'Pa': 91, 'Ra': 88, 'Rn': 86, 'Re': 75, 'Rh': 45, 'Rb': 37,
                   'Ru': 44, 'Rf': 104, 'Sm': 62, 'Sc': 21, 'Sg': 106, 'Se': 34, 'Si': 14, 'Ag': 47, 'Na': 11,
                   'Sr': 38, 'S': 16, 'Ta': 73, 'Tc': 43, 'Te': 52, 'Tb': 65, 'Tl': 81, 'Th': 90, 'Tm': 69,
                   'Sn': 50, 'Ti': 22, 'W': 74, 'U': 92, 'V': 23, 'Xe': 54, 'Yb': 70, 'Y': 39, 'Zn': 30,
                   'Zr': 40}
        xyz = open(file, 'r')
        cartesian = []
        atomnumber = 1
        xyz.readline()
        xyz.readline()
        if xyz.readline()[0].isdigit():
            xyz.seek(0)
            for line in xyz:
                atom, x, y, z = line.split()
                cartesian.append([int(atom), np.array([x, y, z], dtype='f8')])
                atomnumber = atomnumber + 1
        else:
            xyz.seek(0)
            xyz.readline()
            xyz.readline()
            for line in xyz:
                atom, x, y, z = line.split()
                cartesian.append([int(nam2num[atom]), np.array([x, y, z], dtype='f8')])
                atomnumber = atomnumber + 1

        return cartesian

    def read_monomers(file):
        xyz = open(file, 'r')
        xyz.seek(0)
        molecule = []
        molecules = []
        for line in xyz:
            try:
                atom, x, y, z = line.split()
                molecule.append([int(atom), np.array([x, y, z], dtype='f8')])
            except:
                if line != '\n':
                    molecules.append(molecule)
                    molecule = []

        molecules.append(molecule)
        return molecules

    def center_monomers(molecules, outfilename):
        file = open(outfilename, 'w')
        for molecule in molecules:
            center = np.array(center_cartesian(molecule))
            for atom in molecule:
                atom[1] = atom[1] - center

            file.write(f'{len(molecule)}\n')
            file.write(str_cartesian(molecule))

    def str_cartesian(cartesian):
        out = ''
        for atom, coord in cartesian:
            if str(atom).isdigit:
                out += f'{str(atom):<2s} {coord[0]:>15.10f} {coord[1]:>15.10f} {coord[2]:>15.10f}\n'
            else:
                out += f'{str(atom):<2s} {coord[0]:>15.10f} {coord[1]:>15.10f} {coord[2]:>15.10f}\n'

        out += '\n\n'

        return out

    def output_cartesian(coordinates, output_file='testmolecule.xyz'):
        """Output the cartesian coordinates of the file"""
        with open(output_file, 'w') as f:
            for molecule in coordinates:
                f.write(f'{len(molecule)}\n')
                f.write(str_cartesian(molecule))

    def interatomlength(atom1, atom2):
        xyz1 = atom1[1]
        xyz2 = atom2[1]
        currentlength = (m.sqrt(pow((float(xyz1[0]) - float(xyz2[0])), 2) +
                                pow((float(xyz1[1]) - float(xyz2[1])), 2) +
                                pow((float(xyz1[2]) - float(xyz2[2])), 2)))
        return currentlength

    def molecular_delimeter(xyz):
        bondlength = 1.7
        molecule = []
        workspace = []
        item1 = xyz[0]
        i = 0
        workspacecount = 0
        workspace.append(item1)
        del xyz[i]

        for item in xyz:
            currlength = interatomlength(item1, item)
            if item1[0] == 1 or 'H':
                maxcontacts = 2
            elif item1[0] == 6 or 'C':
                maxcontacts = 4
            elif item1[0] == 8 or 'O':
                maxcontacts = 2
            elif item1[0] == 7 or 'N':
                maxcontacts = 3

            if len(workspace) >= maxcontacts:
                break
            else:
                if currlength < bondlength:
                    workspace.append(item)
                    workspacecount += 1
                i += 1

        count = 0
        stoplen = 10
        while len(molecule) != stoplen:
            stoplen = len(molecule)
            for item in workspace:  # remove workspace from xyz
                xyz = [x for x in xyz if x[1].all != item[1].all]
            molecule = molecule + workspace

            # проверить на уникальность перед добавлением
            workspace1 = workspace
            workspace = []
            for item in workspace1:
                for itemxyz in xyz:
                    if interatomlength(itemxyz, item) < bondlength:
                        if len(workspace) > 2:
                            if workspace[-1:][0][1].all != itemxyz[1].all:
                                workspace.append(itemxyz)
                        elif len(workspace) == 1:
                            if workspace[0][1].all != itemxyz[1].all:
                                workspace.append(itemxyz)
                        else:
                            workspace.append(itemxyz)

            i = 0
            for atom1 in workspace:
                j = 0
                for atom2 in workspace:
                    if j > i:
                        if atom1[1].all == atom2[1].all:
                            workspace.pop(j)

                    j += 1
                i += 1
            count = count + 1

        return [molecule, xyz]

    def contactpair(molecule1, molecule2, i, j):
        length = 1000
        for atom1, xyz1 in molecule1:
            for atom2, xyz2 in molecule2:
                currentlength = (m.sqrt(pow((float(xyz1[0]) - float(xyz2[0])), 2) +
                                        pow((float(xyz1[1]) - float(xyz2[1])), 2) +
                                        pow((float(xyz1[2]) - float(xyz2[2])), 2)))

                if currentlength < length:
                    atomout1 = atom1
                    atomout2 = atom2
                    xyzout1 = xyz1
                    xyzout2 = xyz2
                    length = currentlength

        out = [atomout1, xyzout1, atomout2, xyzout2, length, i, j]
        return out

    def dimer_output(dimers, outputFilename='dimers.xyz'):
        f = open(outputFilename, 'w')
        i = 1
        for pair in dimers:
            molecule1 = molecules[pair[5]]
            molecule2 = molecules[pair[6]]
            out = str(len(molecule1) + len(molecule2)) + '\n' + str(pair[4]) + '\n\n'
            dummy = center_cartesian(molecule1 + molecule2)
            dimer = molecule1 + molecule2
            for atom, coord in dimer:
                out += f'{str(atom):<2s} {coord[0] - dummy[0]:>15.10f} {coord[1] - dummy[1]:>15.10f} {coord[2] - dummy[2]:>15.10f}\n'

            i += 1
            out += '\n'
            f.write(out)

        f.close()

    def percent_indicator(current, all):
        percent = int(10000 * (all - current) / all) / 100
        return percent

    def center_cartesian(dimer):
        """Find the center of mass and move it to the origin"""
        masses = {'X': 0, 'Ac': 227.028, 'Al': 26.981539, 'Am': 243, 'Sb': 121.757, 'Ar': 39.948,
                  'As': 74.92159, 'At': 210, 'Ba': 137.327, 'Bk': 247, 'Be': 9.012182, 'Bi': 208.98037,
                  'Bh': 262, 'B': 10.811, 'Br': 79.904, 'Cd': 112.411, 'Ca': 40.078, 'Cf': 251, 'C': 12.011,
                  'Ce': 140.115, 'Cs': 132.90543, 'Cl': 35.4527, 'Cr': 51.9961, 'Co': 58.9332, 'Cu': 63.546,
                  'Cm': 247, 'Db': 262, 'Dy': 162.5, 'Es': 252, 'Er': 167.26, 'Eu': 151.965, 'Fm': 257,
                  'F': 18.9984032, 'Fr': 223, 'Gd': 157.25, 'Ga': 69.723, 'Ge': 72.61, 'Au': 196.96654,
                  'Hf': 178.49, 'Hs': 265, 'He': 4.002602, 'Ho': 164.93032, 'H': 1.00794, 'In': 114.82,
                  'I': 126.90447, 'Ir': 192.22, 'Fe': 55.847, 'Kr': 83.8, 'La': 138.9055, 'Lr': 262,
                  'Pb': 207.2, 'Li': 6.941, 'Lu': 174.967, 'Mg': 24.305, 'Mn': 54.93805,
                  'Mt': 266, 'Md': 258, 'Hg': 200.59, 'Mo': 95.94, 'Nd': 144.24, 'Ne': 20.1797, 'Np': 237.048,
                  'Ni': 58.6934, 'Nb': 92.90638, 'N': 14.00674, 'No': 259, 'Os': 190.2, 'O': 15.9994,
                  'Pd': 106.42, 'P': 30.973762, 'Pt': 195.08, 'Pu': 244, 'Po': 209, 'K': 39.0983,
                  'Pr': 140.90765, 'Pm': 145, 'Pa': 231.0359, 'Ra': 226.025, 'Rn': 222, 'Re': 186.207,
                  'Rh': 102.9055, 'Rb': 85.4678, 'Ru': 101.07, 'Rf': 261, 'Sm': 150.36, 'Sc': 44.95591,
                  'Sg': 263, 'Se': 78.96, 'Si': 28.0855, 'Ag': 107.8682, 'Na': 22.989768, 'Sr': 87.62,
                  'S': 32.066, 'Ta': 180.9479, 'Tc': 98, 'Te': 127.6, 'Tb': 158.92534, 'Tl': 204.3833,
                  'Th': 232.0381, 'Tm': 168.93421, 'Sn': 118.71, 'Ti': 47.88, 'W': 183.85, 'U': 238.0289,
                  'V': 50.9415, 'Xe': 131.29, 'Yb': 173.04, 'Y': 88.90585, 'Zn': 65.39, 'Zr': 91.224}
        num2nam = {89: 'Ac', 13: 'Al', 95: 'Am', 51: 'Sb', 18: 'Ar', 33: 'As', 85: 'At', 56: 'Ba', 97: 'Bk', 4: 'Be',
                   83: 'Bi', 107: 'Bh', 5: 'B', 35: 'Br', 48: 'Cd', 20: 'Ca', 98: 'Cf', 6: 'C', 58: 'Ce',
                   55: 'Cs', 17: 'Cl', 24: 'Cr', 27: 'Co', 29: 'Cu', 96: 'Cm', 105: 'Db', 66: 'Dy', 99: 'Es', 68: 'Er',
                   63: 'Eu', 100: 'Fm', 9: 'F', 87: 'Fr', 64: 'Gd', 31: 'Ga', 32: 'Ge', 79: 'Au', 72: 'Hf', 108: 'Hs',
                   2: 'He', 67: 'Ho', 1: 'H', 49: 'In', 53: 'I', 77: 'Ir', 26: 'Fe', 36: 'Kr', 57: 'La', 103: 'Lr',
                   82: 'Pb', 3: 'Li', 71: 'Lu', 12: 'Mg', 25: 'Mn', 109: 'Mt', 101: 'Md', 80: 'Hg', 42: 'Mo', 60: 'Nd',
                   10: 'Ne', 93: 'Np', 28: 'Ni', 41: 'Nb', 7: 'N', 102: 'No', 76: 'Os', 8: 'O', 46: 'Pd', 15: 'P',
                   78: 'Pt', 94: 'Pu', 84: 'Po', 19: 'K', 59: 'Pr', 61: 'Pm', 91: 'Pa', 88: 'Ra', 86: 'Rn', 75: 'Re',
                   45: 'Rh', 37: 'Rb', 44: 'Ru', 104: 'Rf', 62: 'Sm', 21: 'Sc', 106: 'Sg', 34: 'Se', 14: 'Si', 47: 'Ag',
                   11: 'Na', 38: 'Sr', 16: 'S', 73: 'Ta', 43: 'Tc', 52: 'Te', 65: 'Tb', 81: 'Tl', 90: 'Th', 69: 'Tm',
                   50: 'Sn', 22: 'Ti', 74: 'W', 92: 'U', 23: 'V', 54: 'Xe', 70: 'Yb', 39: 'Y', 30: 'Zn', 40: 'Zr'}
        total_mass = 0.0
        center_of_mass = np.array([0.0, 0.0, 0.0])
        for atom, xyz in dimer:
            # if atom.isdigit():
            mass = masses[num2nam[atom]]
            # else:
            #    mass = masses[atom]

            total_mass += mass
            center_of_mass += xyz * mass
        center_of_mass = center_of_mass / total_mass

        # Translate each atom by the center of mass

        return center_of_mass

    start_time = time.time()
    molecules = []
    cartesian = read_cartesian(inputfile)

    allAmountOfAtoms = len(cartesian)
    #self.ui.statusbar.showMessage('\nAtoms in system: ' + str(allAmountOfAtoms))

    # building monomer
    #self.ui.statusbar.showMessage('Monomers building in process...')
    stoptest = 1
    sortedmolecule = []
    while stoptest:
        molecule = molecular_delimeter(cartesian)
        cartesian = molecule[1]
        sortedmolecule = []
        if len(molecule[0]) > int(minatomsinmol):
            molecules.append(molecule[0])
        stoptest = len(molecule[1])
        #self.ui.statusbar.showMessage('Monomers building in process ' + str(percent_indicator(len(cartesian), allAmountOfAtoms)) + '/100')

    output_cartesian(molecules, output_file=outmonomer)

    molecules = read_monomers(outmonomer)[1:]

    lineprev = ''
    monomer_time = time.time() - start_time
    #self.ui.statusbar.showMessage('\nMonomers building complete by {:<.3} seconds\n'.format(monomer_time))
    #self.ui.statusbar.showMessage('Pairs building in process...')
    # find contact dimers
    dimers = []
    uniquelength = []
    i = 0
    for molecule1 in molecules:
        j = 0
        for molecule2 in molecules:
            if i <= j:
                break
            else:
                dimer = contactpair(molecule1, molecule2, i, j)
                if dimer[4] < int(maxcontact):
                    dimer[4] = int(dimer[4] * 10000) / 10000
                    if dimer[4] not in uniquelength:
                        uniquelength.append(dimer[4])
                        dimers.append(dimer)
            j += 1
        #self.ui.statusbar.showMessage('Pairs building in process ' + str(percent_indicator(len(molecules) - i, len(molecules))) + '/100')
        i += 1

    #self.ui.statusbar.showMessage('Pairs building in process 100.0/100\n')
    dimers.sort(key=lambda i: i[4])
    dimer_output(dimers, outdimer)

    center_monomers(molecules, outmonomer)

    dimer_time = time.time() - start_time - monomer_time
    #self.ui.statusbar.showMessage('Monomers building complete by {:<.3} seconds'.format(monomer_time))
    #self.ui.statusbar.showMessage('Pairs building complete by {:<.3} seconds'.format(dimer_time))
    #self.ui.statusbar.showMessage('Total time: {:<.3} seconds'.format(dimer_time + monomer_time))
