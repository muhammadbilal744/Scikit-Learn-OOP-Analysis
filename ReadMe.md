# Object-Oriented Analysis of Scikit-Learn Library

An exhaustive architectural investigation into the Object-Oriented Programming (OOP) patterns, design principles, and concrete layout structures governing the Scikit-learn framework. This repository features a professional technical report alongside a functional, custom 3-class machine learning extension ecosystem built over standard Scikit-learn interfaces.

---

## Authors (Group Members)
* *Muhammad Bilal* — F25BDATS1M02098
* *Maryam Tahir* — F25BDATS1M02051
* *Jaweria Zafar* — F25BDATS1M02090

* *Submitted to:* Dr. Akmal Shahbaz  
* *Department:* Department of Data Science, The Islamia University of Bahawalpur  
* *Subject:* Object-Oriented Programming (BS Data Science — 2nd Semester — M2 Section)  
* *Session:* 2026

---

## Project Overview

This project bridges the gap between software engineering theory and applied machine learning pipeline creation. It comprises two main artifacts:
1. *The Formal Report (Report.docx):* A granular, code-level analysis tracking line-by-line engineering choices within Scikit-learn's underlying files like base.py, pipeline.py, and _logistic.py.
2. *The Custom Extension (custom_extension.py):* A working, production-grade Python script implementing a custom transformer, classifier, and pipeline ecosystem that inherits from and complies with standard Scikit-learn Mixins.

---

##  Software Architecture & OOP Enforcement

The custom framework developed in custom_extension.py explicitly showcases the four pillars of Object-Oriented Programming:

### 1. Multiple Inheritance & Mixins
Both SmartClassifier and SmartTransformer inherit concurrently from Scikit-Learn's structural layout. 
* SmartClassifier uses BaseEstimator and ClassifierMixin to automatically gain hyperparameter utilities and uniform validation properties.
* SmartTransformer leverages TransformerMixin to acquire automatic fit_transform() behavior through boilerplate code reuse.

### 2. Polymorphism (Method Overriding)
Standard behavioral workflows (fit, predict, transform, score) are overridden to execute custom math patterns while maintaining plug-and-play compatibility with native estimators.

### 3. Abstraction
Granular mathematical manipulations (such as data matrix centering, state constraints tracking via check_array, and multi-class bound logic parsing) are encapsulated away from the terminal client layer behind generic interface loops.

### 4. Object Composition
Rather than using heavy, deep structural inheritance trees, SmartPipeline implements strict *Composition pattern. It acts as an independent execution orchestrator that holds and controls standalone component objects (SmartTransformer and SmartClassifier), respecting the software axiom: *"Favor object composition over class inheritance".

---

##  Repository Blueprint

```text
├── Report.docx            # Detailed Architectural Analysis Document
├── custom_extension.py    # Production-ready Custom Python OOP Implementation
├── .gitignore             # Python environment and metadata filter file
└── README.md              # Project Blueprint & Deployment Manual (This File)
