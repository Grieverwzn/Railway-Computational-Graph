# Computational Graph-based Railway Demand Estimation Toolkit

This script contains functional modules for estimating railway network demand using a computational graph-based approach.  

The input data includes ticket sales data and station information. The output data comprises:  
Estimated boarding passenger volume; OD flows, OD line flows, line section flows, behavioral parameters for the Nested Logit model. The behavioral parameters include frequency coefficients, travel time coefficients, and price coefficients for each line, as well as economic coefficients for each OD pair. The model integrates the Nested Logit model, hierarchical model, and linear regression model, while considering induced demand, diverted demand, and ex-post demand constrained by train seat capacity.  

 <img width="865" height="541" alt="image" src="https://github.com/user-attachments/assets/5402293c-239a-41e2-a833-b665ec8883f1" />

Conceptual illustration of diverted, induced and ex-post demand components.

---

# Getting Started

## 1. Preparing Data  
Prepare a ticket_sale_data.csv file with the following fields:  
1. BOARDDATE: Date of boarding  
2. TRAINID: Unique identifier for the train (corresponds to train number, starting from 0)  
3. TRAINNO: Train number  
4. DEP_STATION_ID: Departure station ID (e.g., A-0, B-1, C-2)  
5. DEP_STATION: Departure station (e.g., A, B, C)  
6. DEP_TIME: Departure time  
7. ARR_STATION_ID: Arrival station ID (e.g., B-1, C-2, D-3)  
8. ARR_STATION: Arrival station (e.g., B, C, D)  
9. ARR_TIME: Arrival time  
10. PSGNUM: Number of passengers (actual tickets sold for the OD segment)  
11. PRICESUM: Total ticket revenue for the OD segment  
12. PRICE: Ticket price per passenger  
13. BASE PSGNUM: Baseline passenger count  

## 2. Main Steps  
The model is trained using the TensorFlow library. The training process involves four interconnected steps:  
1. Training the probability of choosing each OD line using the Nested Logit model  
2. Training flows in the hierarchical structure based on flow conservation principles  
3. Training the OD-based linear regression flow model using coupling constraints  
4. Integrating results from the first three steps to generate the final estimates  

Coupling constraints and capacity constraints are enforced using the augmented Lagrangian method.  


## 3. Output Data  
The tool outputs the following 12 files:  
1. loss_figure.png  
2. nested_logit_statistic_metrics.csv  
3. est_behavior_params.csv  
4. est_board.csv  
5. est_od.csv  
6. est_od_line.csv  
7. est_alight.csv  
8. est_sect_cap.csv  
9. est_sect_flow.csv  
10. r_square.csv  
11. loss.csv  
12. params_setting.csv  

---

## File Descriptions  

### 3.1 loss_figure.png  
Convergence curves of the total loss function and individual loss functions. From left to right and top to bottom: total loss, boarding loss, Nested Logit model loss, alighting loss, section constraint loss, and coupling constraint loss.  

 <img width="865" height="481" alt="image" src="https://github.com/user-attachments/assets/cfd6e154-bdf4-451d-9a4d-e3ee945e13c6" />


### 3.2 nested_logit_statistic_metrics.csv  
   Statistical metrics for the Nested Logit model:  
   - Init Loglikelihood: Initial log-likelihood value  
   - Final Loglikelihood: Final log-likelihood value  
   - Rou Square: Measures improvement over the baseline model (value between 0 and 1; higher values indicate better fit)  
   - AIC: Akaike Information Criterion (lower values indicate better model performance, balancing fit and complexity)  

### 3.3 est_behavior_params.csv 
   Behavioral model parameters, including:  
   - Boarding data  
   - Parameters controlling correlation within nests  
   - Economic impact coefficients for origin stations  
   - Frequency of specific train lines for OD pairs  
   - Travel time of specific train lines for OD pairs  
   - Ticket price of specific train lines for OD pairs  

### 3.4 est_board.csv  
   Boarding data, including:  
   - Boarding time  
   - Departure station ID and name  
   - Total passenger count  
   - Departure station sequence  
   - Estimated passenger count  
   - Data type (training/validation)  

### 3.5 est_od.csv  
   Daily OD data, including:  
   - OD ID (e.g., 0: A-B; 1: A-C; 2: A-D; 3: B-C; 4: B-D; 5: C-D)  
   - Departure and arrival station IDs and names  
   - Station sequence  
   - Passenger count  
   - Economic coefficients  
   - Service frequency  
   - Estimated passenger count  
   - Average estimated passenger count  
   - Data type (training/validation)  

### 3.6 est_od_line.csv  
   OD line data, including:  
   - OD line ID (corresponding to OD line)  
   - Line ID (e.g., 0: A-B-C-D; 1: A-C-D; 2: A-B-D; 3: A-D)  
   - Departure and arrival station IDs, names, and sequences  
   - Total passenger count  
   - Total ticket revenue  
   - Travel time  
   - Ticket price  
   - Observed probability  
   - Service frequency  
   - Average passenger count  
   - Average ticket price  
   - Average travel time  
   - Sections included in the OD line  
   - Estimated passenger count  
   - Estimated probability  
   - Data type (training/validation)  

### 3.7 est_alight.csv  
   Alighting data, including:  
   - Alighting time  
   - Arrival station ID and name  
   - Total passenger count  
   - Arrival station sequence  
   - Estimated passenger count  
   - Data type (training/validation)  

### 3.8 est_sect_cap.csv  
   Line section capacity data, including:  
   - Section ID (corresponding to section name)  
   - Section name (format: "Origin-Destination")  
   - Service frequency  
   - Section capacity  
   - Boarding date  
   - Estimated passenger count  
   - Data type (training/validation)  

### 3.9 est_sect_flow.csv  
   Line section flow data, including:  
   - Boarding date  
   - Section ID (uniquely identifies a railway segment between two adjacent stations)  
   - Section name (composed of station codes at both ends)  
   - Observed section flow (derived from ticket sales data)  
   - Estimated section flow (predicted by the computational graph model)  
   - Passenger count conversion coefficient  
   - Data type (training/validation)  

### 3.10 r_square.csv  
   R-squared values for:  
   - Boarding passenger volume prediction (measures fit between predicted and actual boarding counts)  
   - OD flow prediction (measures fit between predicted and actual OD flows)  
   - OD line flow prediction (measures fit between predicted and actual OD line flows)  
   - Alighting passenger volume prediction (measures fit between predicted and actual alighting counts)  

### 3.11 loss.csv  
   Loss values over iterations, including:  
   - Iteration count (epoch)  
   - Total loss function value (primary optimization target)  
   - Boarding loss (difference between predicted and actual boarding counts)  
   - Nested Logit model loss  
   - Alighting loss (difference between predicted and actual alighting counts)  
   - Section constraint loss (penalty for exceeding train capacity)  
   - OD flow coupling constraint loss  

### 3.12 params_setting.csv  
Configuration parameters for the model.  

## 4.Example: Beijing-Shanghai high-speed railway corridor
A 14-day ticket sales dataset collected from the Beijing-Shanghai high-speed railway corridor serves as a real-world case study to showcase the practical application and impact of the proposed method. The dataset includes two weeks of data: one week before the Spring Festival (January 1st, 2017, to January 7th, 2017) and one week after (February 22nd, 2017, to February 28th, 2017). These periods are selected to minimize the influence of the heightened travel demand typically seen during the Spring Festival in China. 

 <img width="865" height="446" alt="image" src="https://github.com/user-attachments/assets/753f0524-5fd7-4cd8-be4c-a2cc0cd3f7a3" />

The figure shows the line plan operated during the specified period. All train lines operate with a service frequency of 1 from 8:00 a.m. to 10:00 a.m. After shuffling the dataset by days, we use the first 10 days of data as the training set and the subsequent 4 days of data as the validation set. 

<img width="865" height="477" alt="image" src="https://github.com/user-attachments/assets/da6ea073-bfae-4283-a1a6-4beef0f859ed" />

 
This figure further illustrates estimated boarding, alighting and OD demand at stations along Beijing-Shanghai high-speed rail corridor. The travel time, frequency, and ticket price are collected from the real timetable operated at the time.

