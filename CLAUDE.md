- use venv first 
MOTIVATION 
Target users: AI/ML researchers, cloud service providers, policymakers, corporate sustainability teams, and the general public.
Pain point: Lack of intuitive, predictive tools to estimate the energy consumption and carbon footprint of different AI mode
l configurations. People are not aware that each interaction with AI models uses energy and produces a carbon footprint.
(parameter size, hardware, training epochs).   
Value: Provide data-driven guidance for green AI design, supporting ESG reporting and carbon-neutral & sustainable strategies.

  
DATA COLLECTION  
Data sources: MLCO2 Impact, Hugging Face model cards for training energy data; (optional)cloud provider APIs (AWS/Azure/GCP) for cost or energy estimates; academic papers for model parameters, hardware configurations, and training duration.

Management plan: Store all data in structured CSV  tables with metadata fields such as model name, parameter size, training epochs, batch size, hardware type, energy use, and carbon emissions.

PREPROCESSING  
 Cleaning: Remove missing or inconsistent values; standardize energy units.
Transformation: Compute derived features such as energy per parameter and energy per training hour.
Feature engineering: Include hardware type, batch size, and regional carbon intensity; normalize or standardize features.
EXPLORATORY DATA ANALYSIS (EDA)  
 Explore relationships between parameter size and energy/carbon footprint.
Compare average energy use across different GPU/TPU models.
Analyze temporal trends in model energy consumption.

VISUALIZATIONS 

 Interactive scatter plots: parameter size vs. energy/carbon, with filtering by model and hardware.
Bar charts: average energy consumption by hardware type.
Time-series plots: trends in energy usage over time. Build interactive dashboards with Fasthtml or  Streamlit.  LEARNING TASK  (focus on problem definition)
 Supervised regression with target variable: total model energy consumption or carbon emissions.
Input variables: parameter size, training epochs, batch size, hardware type, regional carbon intensity.
Goal: predict energy and carbon footprint for a given model configuration.
 LEARNING APPROACH   (focus on solution implementation)
 Models: multiple linear regression, random forest, XGBoost, lightweight neural networks for comparison.
Evaluation metrics: MAE, RMSE, R².
Validation: K-fold cross-validation or stratified sampling by hardware type.

COMMUNICATION OF RESULTS  

Deliverable: interactive web demo where users input parameters (e.g., parameter size, training duration, hardware type) to instantly get energy and carbon predictions.
The page includes data source and assumption notes, tailored for researchers, policymakers, and sustainability teams.
 
DATA PRIVACY AND ETHICAL CONSIDERATIONS  (if applicable)

Most data are from public research or cloud provider APIs, with low privacy risks.
Verify usage permissions and cite data sources clearly.
Address fairness concerns due to regional variations in carbon intensity.

ADDED VALUE                               

Our ultimate goal is to build a green AI design assistant that helps the user: Quickly estimates model energy consumption and carbon footprint with multi-scenario simulation, while providing optimization advice.
Besides, we provide reusable outputs: quantitative metrics and visualizations for  ESG report generation.
For the general public, our tool makes the hidden carbon cost of AI interactions visible, raising awareness of their environmental  impact.
*Continuous updates(in the future): Regularly fetch the latest model energy data and grid carbon intensity to keep predictions current.
