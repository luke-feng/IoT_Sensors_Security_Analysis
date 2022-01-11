# IoT_Sensors_Security_Analysis
> A pricavy-preserving SSDF cyberattack intelligent detection system.
## Table of Contents

   * [Introduction](#introduction)
   * [Sensor](#sensor)
      * [System calls monitoring](#system-calls-monitoring)
      * [Run in Background](#run-in-Background)
   * [Data Moudle](#data-moudle)
      * [Data Cleaning](#data-cleaning)
      * [Feature Extracion](#feature-extracion)
      * [Normalization](#normalization)
      * [Get PCA](#get-pca)
   * [Detection Module](#detection-module)
      * [Machine Learning Based Approaches](#machine-learning-based-approaches)
      * [Federated Learning Based Approaches](#federated-learning-based-approaches)

## Introduction
> Over the past few decades, IoT technologies have surged, with billions of devices accessing the Internet through wireless networks, bringing convenience to human lives while consuming the valuable wireless spectrum. To optimize the radio frequency spectrum, crowdsensing-based radio frequency spectrum monitoring networks are proposed, consisting of distributed IoT sensors that collaborate to collect, transmit, and process radio spectrum data worldwide. However, these IoT sensors with constrained resources are extremely vulnerable to cyberattacks that compromise the integrity of the radio frequency spectrum data and affect the operation of the entire platform. 

> On the one hand, Machine Learning-based device behavior fingerprinting for cyberattack identification is considered highly promising. On the other hand, the device behavior data is strongly sensitive, and its data privacy becomes an issue that has to be considered. Taking these into consideration, this thesis proposes a Federated Learning-based IoT network attack detection system using system calls behavioral data. This approach achieves both data privacy protection and effective identification of cyberattacks through its unique training strategy, i.e., sharing only model parameters but not the training data. After a systematic comparison, this thesis selects the most suitable feature extraction approach and local identification algorithm. The effectiveness and reliability of the proposed model is demonstrated by using quantitative analysis through a variety of different scenarios.

The following shows the outline structure of the this system.
   ```bash
      |—— sensor
        |—— get_system_perf.sh
        |—— monitoring.sh
        |—— run_background .sh
      |—— datamodule
        |—— preprocessing_pref.py
        |—— get_features.py
        |—— normalization.py
        |—— get_pac.py
      |—— detectionmodule
        |—— ml
        |—— fl
   ```
   
   
