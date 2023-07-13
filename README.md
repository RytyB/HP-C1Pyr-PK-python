<h1> HP C1-Pyruvate Pharmacokinetic Modeling Package for Python </h1>

<h2>
Purpose
</h2>

<ul>
This package was meant to create a more user friendly, open source version of the MATLAB developed by the MD Anderson Magnetic Resonance Engineering Laboratory.
The original repository is available at https://github.com/mda-mrsl/HP-C1Pyr-PK
</ul>

<ul>
These scripts calculate the relationship between [1-13C]-pyruvate and [1-13C]-lactate, but can easily be extended to include other metabolites.
</ul>

<ul>
There are three base models, of differing complexity, which include a different number of physical compartments:
</ul>

<ul>
The scripts titled P2L3*.py describe signal evolution between two chemical pools and three physical compartments. The three-compartment model is the most physiologically accurate among these models, but it is also the most computationally intensive, requiring the largest number of descriptive parameters.
</ul>

<ul>
The scripts titled P2L2*.py are for the PK model with two chemical pools and two physical compartments. Here, HP pyruvate and lactate in the extravascular/extracellular space is assumed to be very well mixed with HP pyruvate in the intracellular space - and separate from pyr/lac in the vascular supply.
</ul>

<ul>
The scripts titled P2L1*.py represent a simple precursor-product model, which does not consider physical compartmentalization of imaging agents. Equivalently, this assumes that all compartments (vascular; extravascular/extracellular; and cellular) are well mixed.
</ul>

<ul>
Each script is accompanied by utilities and a test function which runs a model and plots the results for given parameters. 
</ul>

<h2> 
Installation
</h2>

<ul> Running requires python version 3.10 or compatible </ul>

<ul>
Can be installed from Pypi repository with pip
<br> <i> pip install pkhpc1pyr </i>
<br> The required dependencies can be found in the file requirements.txt and installed with the command 
<br> <i> pip install numpy==1.24.3 </i>
</ul>

<ul>
The installation can be tested by calling the TestP2L* functions without any arguments. For example:
<br> <i> import pkhpc1pyr as pk 
<br> pk.tests.TestP2L3() </i>
</ul>

<h2> 
Contact
</h2>

<ul>
For questions about code or maintenance please contact Ryan Boyce <br> rboyce@mdanderson.org
<br> For further questions about the original MATLAB repository please contact Dr. Jim Bankson <br> jbankson@mdanderson.org
</ul>

