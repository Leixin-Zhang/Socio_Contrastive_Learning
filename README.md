
## Modeling Human Perspectives with Socio-Demographic Representations

<div align="center",style="font-family: charter;">
    <a href="https://scholar.google.com/citations?user=dTRy2gUAAAAJ&hl=en" target="_blank">Leixin Zhang</a>, 
    <a href="https://coltekin.net/cagri/" target="_blank">Çağrı Çöltekin</a>
</div>


## Code Structure for the Project:
```
Socio_Contrastive_Learning
│
├── data_processing/
│   ├── hatespeech_data_processing.py
│   ├── toxicity_data_processing.py
│   ├── dataset_loader.py
│   └── text_encoder.py
│
├── models/
│   ├── baseline_model.py
│   ├── socio_feature_model.py
│   └── contrastive_model.py
│
├── training/
│   ├── self_defined_loss.py
│   ├── trainer_classes.py
│   └── train_models.py
│
├── evaluation/  
│   └── evaluators.py
│   
└── run_all_models.py
```


## 🚀 Introduction: 

**Background:** Humans often hold different perspectives on the same issues. Modeling annotator perspectives and understanding their relationship with other human factors have received increasing attention. In real-world settings, annotator perspectives are shaped by complex social contexts. However, prior work typically focuses on individual demographic factors or limited combinations. 

**Contribution:** 🏆 Socio-Contrastive Learning, a method that jointly models annotator perspectives while learning socio-demographic representations from a set of socio-demographic features. 

**Advantages:** An effective approach for the fusion of socio-demographic features and textual representations to predict annotator perspectives. The learned representations further enable analysis and visualization of how demographic factors relate to variation in annotator perspectives.

## 💡 Visualization: 

**Annotator Representation for the Hate Speech Dataset**
![Annotator Representation for Hate Speech Dataset](images/Annotator_Representation_Hatespeech_1_page-0001.jpg)
![Annotator Representation for Hate Speech Dataset](images/Annotator_Representation_Hatespeech_2_page-0001.jpg)
**Annotator Representation for the Toxic Dataset**
![Annotator Representation for Toxic Dataset](images/Annotator_Representation_Toxic_1_page-0001.jpg)
![Annotator Representation for Toxic Dataset](images/Annotator_Representation_Toxic_2_page-0001.jpg)
