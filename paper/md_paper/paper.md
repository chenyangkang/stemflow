---
title: 'stemflow: A Python Package for Adaptive Spatio-Temporal Exploratory Model'
tags:
 - Python
 - ecology
 - spatio-temporal big data
 - citizen science
 - modeling
 - seasonal migration
authors:
 - given-names: Yangkang
   surname: Chen
   orchid: 0000-0003-4996-0881
   affiliation: "1, 2"
 - given-names: Zhongru
   surname: Gu
   orchid: 0000-0002-0178-1107
   affiliation: "1, 3"
 - given-names: Xiangjiang
   surname: Zhan
   corresponding: true
   orchid: 0000-0002-4517-1626
   affiliation: "1, 2, 3, 4"

affiliations:
 - name: Key Laboratory of Animal Ecology and Conservation Biology, Institute of Zoology, Chinese Academy of Sciences, Beijing, China
   index: 1

 - name: University of Chinese Academy of Sciences, Beijing, China
   index: 2

 - name: Cardiff University-Institute of Zoology Joint Laboratory for Biocomplexity Research, Chinese Academy of Sciences, Beijing, China
   index: 3

 - name: Center for Excellence in Animal Evolution and Genetics, Chinese Academy of Sciences, Kunming, China
   index: 4

date: 20 September 2023
bibliography: "paper.bib"

---

# Summary

Stemflow is a user-friendly Python package for Adaptive Spatio-Temporal Exploratory Model (AdaSTEM [@fink_adaptive_2013]) that follows the style of scikit-learn BaseEstimator class [@pedregosa2011scikit]. It provides one-line model creation, fitting, prediction, and evaluation. It implements spatio-temporal train-test-split and cross-validation functions. After model training, feature importance could be evaluated with spatio-temporal dynamics. Stemflow also provides functions for visualizing ensembles structured in model training and generating GIF file for predicted results to animate the spatio-temporal movement of animal population.

# Statement of need

Spatio-temporal big data is an emerging but valuable resource for ecological studies as human enter the era of big data [@farley_situating_2018]. Data from broad-scale survey like citizen science projects is increasingly important in modern ecological research [@dickinson_citizen_2010]. Intensity of survey activities grows rapidly as more people are involved in citizen science in recent years, resulted in exponential accumulation of observational data [@di_cecco_observing_2021; @sullivan_ebird_2014]. However, daily species observation records uploaded by non-professionals in citizen science program are known to have larger bias than professionally structured research, both in terms of data veracity and spatio-temporal balance of the datasets, which necessitates elaborate modeling methods to mine its merits [@dickinson_citizen_2010; @farley_situating_2018].

Some species distribution modeling (SDM) approaches were brought forward to adjust for bias in citizen science and model on the unobserved components [@bird_statistical_2014]. Still, many failed to account for the autocorrelation of space and time [@f_dormann_methods_2007], which is especially crucial in modeling inherently spatio-temporal biological events with variations at different scales [@chave_problem_2013; @levin_problem_1992], such as seasonal migration. Adaptive Spatio-Temporal Exploratory Model (AdaSTEM) is a semi-parameterized machine learning model that leverages the spatio-temporal adjacency information of sample points to model occurrence or abundance of species [@fink_adaptive_2013]. A QuadTree algorithm [@samet_quadtree_1984] is implemented to split data into smaller spatio-temporal grids (called stixels) conditional on the data abundance, with more abundant data allowing stixels to be divided into finer resolution (up to a maximum). Stixels with data size less than a certain threshold will not be modeled; instead, these stixels will be labeled as unpredictable. This procedure controls the degree of model extrapolation and reduces overfitting. A base model is trained for each stixel, that is, targets are only modeled on their adjacent information in space and time. Splitting-training is carried out several times to generate multiple ensembles. Finally, prediction results were aggregated across these ensembles.

AdaSTEM shows the capacity of supporting large scale spatio-temporal ecological data modeling in many studies [@fink_modeling_2020; @fuentes2023birdflow;@la_sorte_seasonal_2022], espetially for modeling animal abundance at different scales [@fink_adaptive_2013]. One well-known application of AdaSTEM is the weekly abundance map of eBird Status and Trend product [@FinkStatusTrend2022], which was widely used as data sources of abundance data of bird populations [@bird_statistical_2014; @jarzyna_decoupled_2023; @lin_using_2022]. The application of AdaSTEM could be extended to other fields with similar data structure and spatio-temporal dependence, for example, epidemiology. Despite the foreseeable significant role of spatio-temporal big data in the coming decades of scientific research, the development of tools has not necessarily kept pace.

Stemflow is positioned as a user-friendly Python package to meet the need of general application of modeling spatio-temporal large datasets. Scikit-learn style object-oriented modeling pipeline enables concise model construction with compact parameterization at the user end, while the rest of the modeling procedures are carried out under the hood. Once the fitting method is called, the model class recursively splits the input training data into smaller spatio-temporal stixels using QuadTree algorithm. For each of the stixels, a base model is trained only using data falls into that stixel. Stixels are then aggregated and constitute an ensemble. In the prediction phase, stemflow queries stixels for the input data according to their spatial and temporal index, followed by corresponding base model prediction. Finally, prediction results are aggregated across ensembles to generate robust estimations (see @fink_adaptive_2013 and stemflow documentation for details).

For survey projects that include abundance information like eBird [@sullivan_ebird_2014], the targeted modeling values are often zero-inflated, owning to the fact of low observation probability in many species. Zero-inflation could lead to poor regression model performance [@campbell_consequences_2021]. In stemflow, we implement hurdle model classes that embed two sequential models: a classifier to classify the absence and presence states, followed by a regressor to model the abundance for prediction samples classified as presence. Hurdle model classes can be conjunctively used with AdaSTEM model classes in two ways: Use hurdle model as the base model for AdaSTEMRegressor (as in @johnston_abundance_2015), or use AdaSTEMClassifier and AdaSTEMRegressor as the classifier and regressor in hurdle model. We demonstrate the comparison of these two architectures in stemflow documentation.

One advantage of applying stemflow in scikit-learn style is that there is a variety of “base models” to choose from scikit-learn or scikit-learn-style repertoire. The choices vary from linear models to boosting and bagging tree-based models. Maxent model [@anderson_elapid_2023] is also supported to play the role of “base model”, which largely expands the potential application for presence-only modeling (see documentation).

While there exists mounting open source packages for species distribution modeling (mostly in R, [@norberg_comprehensive_2019]; and one in Python [@anderson_elapid_2023]), most of them solely leverage environmental variables and do not support integration of spatio-temporal information during model construction (but see @anderson_elapid_2023; @anderson_sdmtmb_2022; @dobson_span_2023). This disadvantage is usually noted along with the overconfidence of the model extrapolation capacity both for Maxent-based and ensemble-based models [@a_leeyaw_species_2022]. To our knowledge, stemflow is the first package designed to solve the spatio-temporal dependency conjugated with bias in data abundance distribution in species distribution modeling. With the rapid accumulation of data and development of machine learning techniques, stemflow will show potentials in spatio-temporal modeling, and could be applied to other fields (e.g., epidemiology and weather prediction) in future.


# Acknowledgements

This project was based upon work supported by National Natural Science Foundation of China grants 32125005 (to X.Z.), 32270455 (to Z.G.), CAS Project for Young Scientists in Basic Research (YSBR-097 to X.Z. and Z.G.). We thank Dr. Daniel Fink at Cornell Lab of Ornithology for the R scripts of STEM model and QuadTree which inspired our implementation in Python.

# References
