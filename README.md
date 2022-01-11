# IoT_Sensors_Security_Analysis
> A pricavy-preserving SSDF cyberattack intelligent detection system.
## Table of Contents

   * [Introduction](#introduction)
   * [Sensor](#sensor)
      * [System calls monitoring](#system-calls-monitoring)
      * [Run in Background](#run-in-Background)
   * [Data Module](#data-module)
   * [Detection Module](#detection-module)
      * [Machine Learning Based Approaches](#machine-learning-based-approaches)
      * [Federated Learning Based Approaches](#federated-learning-based-approaches)

# Introduction
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
# Sensor
   The sensor folder contains all required scripts for monitoring the system calls from the Raspberry Pis.
   * First you need to install the Perf tool to your Raspberry Pis:
   
    
    sudo apt-get install perf
    

   * Then you can test whether it has been successfully installed via typing  `perf `, if it not works, then you need to change the exec of perf by:

    
    sudo nano /usr/bin/perf
    #exec "perf_$version" "$@"
    exec "perf_4.9" "$@"

## System calls monitoring
   * `get_system_perf.sh`is responding for finding the desired PID, and monitoring it system calls. There are several parameters that you can modify to adopt your tasks.
      * `time_window` means how many seconds that you want to monitor the process each time, by default it's 60s.
      * `total_loop` means how many loops that you can to monitor this process, by default it's 360 loops.
   * `monitoring.sh` is responding for copy the attack files to the service, and automaticly start the get_system_perf service to monitor the system calls of attack/normal services.
## Run in Background
   * `run_background.sh' could run this system calls monitoring script in background. 
   
    ./run_background.sh
   
# Data Module
 > Data module is used to extract the features from the monitored system call files.
   * `preprocessing_pref.py` is used to remove the non-relevant data from the raw data, and keep the system calls only.
   * `get_features.py` is used to convent the raw data to the features, and several feature extraction approaches are used, including the frequency, TF-IDF, One-hot, Dict-index, Dependency-graph.
   * `Normalization.py` is used to normalize the features to [0,1] space.
   * `get_pca.py` is used to fit and transform the PCA from high-dimension features.
# Detection Module
> Detection module is responding for using AI techs to identify the cyberattacks from the data.
## Machine Learning Based Approaches
   * Four ML-based and one DL-based algorithms are used in ML-based approaches, including: One-class SVM, Isolation Forest, Robust Covariance, SGD One-class SVM, as well as the Autoencoder.
   * `ml-based.py` is used to train and test the ML-based models.
   * `dl-based.py` is used to train and test the DL-based models.
## Federated Learning Based Approaches
