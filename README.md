# Introduction to Machine Learning Contest


### Task

The mission of the group is to use (supervised) learning techniques to design a model able to predict the activity of a chemical compound (described by its molecular structure).

### Problem Statement
The original study was about determining the ability for a chemical compound to inhibit HIV replication. Therefore, tens of thousands of compounds have been checked for evidence of anti-HIV activity.
The dataset is made of two components:
- chemical structural data on compounds: each chemical compound is described under the SMILES format. SMILES, standing for Simplified Molecular Input Line Entry Specification, is a line notation for the chemical structure of molecules.
- HIV-activity : it corresponds to the screening result evaluating the activity (1) or the inactivity (0) of the chemical compound.

### Toolkit for cheminformatics RDKit
In order to generate features from SMILES, you may employ the open source toolkit for cheminformatics RDKit.
In particular, you may need the installation steps, see [How to install RDKit](https://rdkit.readthedocs.io/en/latest/Install.html#how-to-install-rdkit-with-conda).
An example of how to generate features with this toolkit is given in the toy_example.py script.

More info [here](https://www.kaggle.com/c/iml2019/overview).

