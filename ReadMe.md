# Object-Oriented Analysis of Scikit-Learn Library

This project presents an Object-Oriented Programming (OOP) analysis of the Scikit-learn library. The main purpose of this project is to understand how Scikit-learn uses important OOP concepts such as inheritance, polymorphism, abstraction, and composition in its internal architecture. This repository also includes a custom machine learning extension built using standard Scikit-learn interfaces.

---

## Authors (Group Members)

* **Muhammad Bilal** — F25BDATS1M02098
* **Maryam Tahir** — F25BDATS1M02051
* **Jaweria Zafar** — F25BDATS1M02090

**Submitted to:** Dr. Akmal Shahbaz  
**Department:** Department of Data Science, The Islamia University of Bahawalpur  
**Subject:** Object-Oriented Programming (BS Data Science — 2nd Semester — M2 Section)  
**Session:** 2026  

---

## Project Overview

This project contains two main parts:

1. **Formal Report:**  
   A detailed analysis of Scikit-learn source code files such as `base.py`, `pipeline.py`, and `_logistic.py`. The report explains how OOP concepts are implemented inside the library.

2. **Custom Extension:**  
   A custom Python implementation containing a transformer, classifier, and pipeline system built using Scikit-learn Mixins and Base Classes.

---

## Software Architecture & OOP Concepts

The custom framework developed in this project demonstrates the four main pillars of Object-Oriented Programming.

### 1. Inheritance & Mixins

Both `SmartClassifier` and `SmartTransformer` inherit from Scikit-learn base classes.

* `SmartClassifier` uses `BaseEstimator` and `ClassifierMixin`.
* `SmartTransformer` uses `TransformerMixin` to get automatic `fit_transform()` functionality.

This shows how Scikit-learn reuses code using inheritance and mixins.

---

### 2. Polymorphism

Methods such as `fit()`, `predict()`, `transform()`, and `score()` are overridden with custom behavior while still following the standard Scikit-learn workflow.

---

### 3. Abstraction

Complex internal operations such as validation and data processing are hidden behind simple method interfaces. This makes the library easier to use.

---

### 4. Composition

`SmartPipeline` follows the Composition pattern by combining separate objects like `SmartTransformer` and `SmartClassifier` instead of creating a deep inheritance structure.

This follows the software design principle:

*"Favor object composition over inheritance."*

---

## Repository Structure

```text
SCIKIT-LEARN-OOP-ANALYSIS/
│
├── analyzed_files/       # Files used during analysis
├── Code/                 # Custom Python implementation
├── diagrams/             # UML and architecture diagrams
├── Report/               # Final academic report
│
├── .gitignore            # Python environment filter file
└── README.md             # Project documentation
