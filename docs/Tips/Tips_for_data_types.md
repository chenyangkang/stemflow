# Tips for data types

We illustrate the data types that are expected to be fed into `stemflow`


## Continuous and categorical features

Basically, `stemflow` is a framework for spatial temporal indexing during modeling. It serves as a container to help `base model` do better jobs, and prevent distant modeling/prediction problem in space and time. Therefore, any feature you use during common tabular data modeling could be used here. It means that both continuous and categorical features can be the input, based on your expectation in the feature engineering.

For categorical features, we recommend [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) if the size of the category is not too large.

Tree-based models (e.g., decision tree, boosting tree, random forest) are robust to missing values so you can fill the missing values with artificial values like `-1`. For other methods, there are different ways to fill the missing values, with [pros and cons](https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e).

------
## Static and dynamic features

### Concepts and examples
Static features are summaries of status, while dynamic features vary largely across each almost each records.

For example, for a task modeling the abundance of a bird each day in 2020;

Static features:

    - Land cover of year 2020:
        - percentage cover of forest
        - patch density of urban area

    - Climate of year 2020:
        - BIO1: Annual Mean Temperature
        - BIO2: Mean Diurnal Range (Mean of monthly (max temp - min temp))
        - BIO19: Precipitation of Coldest Quarter

    - Normalized difference vegetation index (NDVI):
        - NDVI_max: Highest NDVI of the annual cycle
        - NDVI_std: Variation of NDVI in the annual cycle
    

Dynamic features:

    - Weather of each checklist (record):
        - the temperature of the hour (that we observed this bird)
        - the total precipitation of the hour (that we observed this bird)
        - V component of wind of the hour (that we observed this bird)

    - Normalized difference vegetation index (NDVI):
        - The absolute NDVI of the day (that we observed this bird)


### Use of static and dynamic features

Although all features except `DOY` in our [mini test](https://chenyangkang.github.io/stemflow/Examples/00.Mini_test.html) and [AdaSTEM Demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html) are static features, the model fully support dynamic feature input.

Noteworthy, the choice of static or dynamic features depends on some aspects:

1. **Model assumption**: Does the target value vary in response to static summaries or agilely in response to realtime changes?
1. **Scale of interest**: Are you interested in overall smoothed pattern or zig-zag chaotic pattern?
1. **Caution for overfitting**: `stemflow` splits data into smaller spatiotemporal grids. It may induce local overfitting to some extent. By using dynamic features, you should be additionally cautious for overfitting in the scale of time.
1. **Anchor the prediction set**: Make sure you use the same dynamic variables in your prediction set if they are used to train the model. This may cause additional computational challenges.

Likewise, we use static features for several reasons:

1. In our demonstration, static features are used as "geographical configuration". In other words, we are interested in **how birds choose different types of land according to the season**. These static features are highly summarized and have good representation for biogeographic properties.
1. We are interested in large-scale season pattern of bird migration, and are not interested in transient variation like hourly weather.
1. Keeping only `DOY` as dynamic features (temporal variables) reduces the work in compiling a prediction set. Instead of making a realtime one, now we only need to change DOY (by adding one each time) and feed it to `stemflow`. It also reduces memory/IO use.

We recommend users thinking carefully before choosing appropriate features, considering the questions above and availability of computational resources.
