# Development of a Multi-modal CYP2D6 Variant-Drug Metabolic Activity Predictor via a Foundation Model-based Approach
Official repository of cypvar-esm study

# Abstract
Background and Purpose
Genetic variations in cytochrome P450 enzymes play a major role in pharmacogenomic studies. CYP2D6 is a highly polymorphic isoform responsible for metabolising approximately 20–25% of clinically prescribed drugs. However, the systematic prediction of variant-drug metabolic activity remains limited due to heterogeneous, inconsistent, and limited datasets. We aimed to generate a gold-standard experimental dataset and establish a modelling strategy capable of accurately capturing variant-drug interactions.
Experimental Approach
We experimentally measured the metabolic activity of nine clinically relevant CYP2D6 alleles across 35 drugs under uniform conditions to create a gold-standard dataset. To leverage this resource, a deep learning prediction model was developed based on foundation models to capture local interactions between CYP2D6 variants and drug molecules. A protein foundation model of CYP450 sequence variants was fine-tuned to improve variant-specific sensitivity, and the resulting representations were integrated with molecular embeddings. These were combined through a cross-attention mechanism, enabling the explicit modelling of variant-drug interactions. This approach was rigorously evaluated against various baseline models.
Key Results
The generated gold-standard dataset provided a robust and consistent foundation for predictive modelling, demonstrating improved performance over conventional machine-learning-based models. Attention analysis revealed mechanistic interpretability, highlighting how variant and drug features were differentially weighted when predicting metabolic activity.
Conclusion and Implications
This study provides a gold-standard dataset and a novel predictive framework for CYP2D6 variant-drug metabolism. By combining curated experimental evidence with foundation model-based strategies, we improved pharmacogenomic predictions in low-data regimes and provided an interpretable framework for precision medicine and safer drug development.



# Data availiability
Data and model weight will be provided soon.
