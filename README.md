# Introduction

This is the implementation of the *Non-parametric Learning of Lifted Restricted Boltzmann Machines* (LRBM-Boost) paper which is available at [this link](https://www.sciencedirect.com/science/article/pii/S0888613X19302749).

*LRBM-Boost* is first learning algorithm for learning the structure and the parameters of lifted Restricted Boltzmann Machines(RBM) from data. Motivated by the success of functional gradient-boosting, this method learns a set of Relational Regression Trees using boosting and then transforms them to a lifted RBM.

This code builds upon the popular Boost-SRL tool available here [this link](https://github.com/starling-lab/BoostSRL) and the detailed documentaion is available [here](https://starling-lab.github.io/software/boostsrl/) and the type of input needed to run LRBM-Boost is identical to one required for running Boost-SRL tool.

# Software Requirement

## Prerequisites:

Java (tested with openjdk 1.8.0_144)

## Main file

The main file to run this code is *RRBMBoostWithdIncluded/src/edu/wisc/cs/will/Boosting/RBM/RunBoostedRBM.java*

## Basic Usage

![Alt text](https://github.com/navdeepkjohal/LRBM-Boost/blob/master/Image/basicFileStructure.png)

*The above image is borrowed from [BoostSRL](https://github.com/starling-lab/BoostSRL)

The training data required to run *LRBM-Boost* requires:

* *train_pos.txt*: the positive examples required to train the model.

* *train_neg.txt*: the negative examples required to train the model.

* *train_facts.txt*: file containing the facts that will form body of the rules.

* *train_bk.txt*: the file containing the modes required to learn the rules. (To learn how to set the modes, follow [this](https://starling-lab.github.io/software/boostsrl/wiki/basic-modes/) tutorial.)

BoostSRL assumes that data are contained in files with data structured in predicate-logic format.

### Positive Examples

```prolog
father(harrypotter,jamespotter).
father(ginnyweasley,arthurweasley).
father(ronweasley,arthurweasley).
...
```

### Negative Examples

```prolog
father(harrypotter,mollyweasley).
father(harrypotter,lilypotter).
father(harrypotter,ronweasley).
...
```

### Facts

```prolog
male(harrypotter).
male(jamespotter).
siblingof(ronweasley,fredweasley).
siblingof(ronweasley,georgeweasley).
childof(jamespotter,harrypotter).
childof(lilypotter,harrypotter).
...
```

## Learning a Lifted Restricted Boltzmann Machines:

```bash
cd RRBMBoostWithdIncluded
```

```bash
java edu.wisc.cs.will.Boosting.RBM.RunBoostedRBM -rbm -l -train train/ -target father -trees 10
```
## Inference with the Lifted Restricted Boltzmann Machines:

```bash
java edu.wisc.cs.will.Boosting.RBM.RunBoostedRBM -rbm -i -model train/models/ -test test/ -target father -trees 10
```
### Cite

```bash
@article{LRBMBoostKAUR2020,
title = {Non-parametric learning of lifted Restricted Boltzmann Machines},
journal = {International Journal of Approximate Reasoning},
volume = {120},
pages = {33-47},
year = {2020},
issn = {0888-613X},
doi = {https://doi.org/10.1016/j.ijar.2020.01.003},
url = {https://www.sciencedirect.com/science/article/pii/S0888613X19302749},
author = {Navdeep Kaur and Gautam Kunapuli and Sriraam Natarajan},
keywords = {Restricted Boltzmann Machines, Learning lifted models, Functional gradient boosting}
}
```
