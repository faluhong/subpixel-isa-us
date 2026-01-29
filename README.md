# Subpixel impervious surface area across the contiguous U.S. (CONUS)

## Overview
This repository contains data and code related to the estimation of subpixel 
impervious surface area across (%ISA) the contiguous United States (CONUS).

The paper is under preparation and the dataset will be made publicly available upon publication.

The current repository is mainly for peer review purposes. 

## Code Explanation
The repository contains the scripts, auxiliary data and 
statistical results related to the CONUS %ISA dataset generation and ISA analysis.

```text
subpixel-isa-us/
├── data/           # Auxiliary datasets for the production generation
├── pythoncode/     # Core scripts for data processing, analysis and visualization
├── results/        # Statistical results for accuracy assessment, analysis and visualization
├── LICENSE         # License file
└── README.md       # Project documentation
```

The _**pythoncode**_ folder contains the main code for data processing, analysis and visualization, written with Python 3.10 and Google Earth Engine.
- **_high_resolution_land_cover_process_**: Process high-resolution land cover datasets as training sample
- **_model_training_**: Train the U-Net model for %ISA estimation
- **_conus_isa_production_**: Generate the CONUS %ISA product on HPC environment
- **_post_processing_**: Post-processing the original %ISA estimation results to generate the final %ISA product
- **_accuracy_assessment_**: Accuracy assessment of %ISA estimation and IS change detection
- **_conus_isa_analysis_**: Analyze the spatial and temporal patterns of %ISA and its changes
- **_conus_isa_financial_crisis_resilience_**: Analyze the reduction and recovery of %ISA 
- **_conus_isa_centroid_**: Analyze the centroid shift of %ISA
- **_conus_isa_socio_economic_**: Analyze the %ISA changes with socio-economic metrics changes
- **_util_function_**: Utility functions used across different scripts
- **_gee_app_**: Google Earth Engine app for visualizing the %ISA product: https://gers.users.earthengine.app/view/conus-isa


## Contact
For questions or further information, please contact:
- Falu Hong (faluhong@uconn.edu)
- Zhe Zhu (zhe@uconn.edu)
