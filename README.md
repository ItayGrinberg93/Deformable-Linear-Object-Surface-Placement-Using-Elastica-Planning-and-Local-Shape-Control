
# Deformable-Linear-Object-Surface-Placement-Using-Elastica-Planning-and-Local-Shape-Control-based-Vision
This repository contains two-layered DLO placement method, high-planner based on Euler’s analytic elastica solutions and low-level controller that estimates the DLO current shape using Residual Neural Networks.


<!--video of simulation-->

<p align="center">
<b><i>The placement process</i></b>
</p>


<p align="center">
<video src="
" alt="" width="33%">
</p>





<p align="center">
<b>Itay Grinberg & Aharon Levin</b>
<br>
<a href="mailto:itaygrinberg@campus.technion.ac.il" target="_top">itaygrinberg@campus.technion.ac.il</a>
</p>
<p align="center">
<a href="mailto:aharon.levin@campus.technion.ac.il" target="_top">aharon.levin@campus.technion.ac.il</a>
</p>

<p align="center">
<b>Guidance: Elon Rimon</b>
<br>
<a href="mailto:rimon@me.technion.ac.il" target="_top">rimon@me.technion.ac.il</a>
</p>


video: https://youtu.be/li87gRbQ-7c

arxiv: https://doi.org/10.48550/arXiv.2503.08545



------------

<a id="top"></a>
### Contents
1. [Introduction](#1.0)
2. [Environment Setup + Code Documentation](#2.0)

------------

### Abbreviations
* **DLO** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Deformable Linear Object](https://en.wikipedia.org/wiki/Soft-body_dynamics)

------------

<a name="1.0"></a>
### 1. Introduction

The proposed repository focuses on a two-layered approach for placing DLOs on a flat surface using a single robot hand.  The high-level layer is a novel DLO surface placement method based on Euler's elastica solutions. 
The low-level layer forms a pipeline controller. The controller estimates the DLO current shape using a Residual Neural Network (ResNet) and uses feedback to ensure task execution in the presence of modeling and placement errors.  

##### Objective

In order to place the DLO effectively, precise manner and without damage or distortions, the need to make a controlled and well-planned movement.
Our work proposed a novel approach to shape control of DLOs which could have potential applications in the handling of fresh food prodacts.
This is a challenging problem since it requires an online analysis of the current setting of the environment and the current capabilities of the agent. In addition, it requires an integration of high-level task planning and low-level motion control. 

Within the context of this, the placement cycle can be divided into the following tasks:

* Bring the DLO from the transportation process target to the attachment point with the tray.
* Getting the DLO ready for rolling, the tip does not slip and the tray does not slip on the worktable.
* DLO pure rolling until it is placed.

##### Relevance
Manipulation of DLO is relevant to both industrial and household environments. 



------------
<a name="2.0"></a>
<!--<div style="text-align:left;">
  <span style="font-size: 1.4em; margin-top: 0.83em; margin-bottom: 0.83em; margin-left: 0; margin-right: 0; font-weight: bold;"> 2. Environment Setup</span><span style="float:right;"><a href="#top">Back to Top</a></span>
</div>-->
### 2. Environment Setup + Code Documentation
#### 2.1 Environment Setup
The project uses ROS2 Humble Hawksbill running on Ubuntu 22.04 LTS (Jammy Jellyfish) and Python 3.10.


#### 2.2 Code Documentation

The two-layered framework code is splits to high-palnner and low-level controller. 

##### high-level placment planning

##### low-level placment control


------------


