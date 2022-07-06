# Example Notebooks

* [spanner IGW](spannerIGW.ipynb)
    * [ibid but with tuned context embedding](spannerIGWdynamic.ipynb)
* [spanner epsilon-greedy](spannerepsgreedy.ipynb)
* [supervised baseline](supervised.ipynb)
    * [ibid but with tuned context embedding](superviseddynamic.ipynb)
* [IGW baseline](IGW.ipynb)
    * [ibid but with tuned context embedding](IGWdynamic.ipynb)

# How to get the data

* download from https://www.kaggle.com/generall/oneshotwikilinks
* unzip
* run `cut -f1 shuffled_dedup_entities.tsv | sort -S50% | uniq -c | sort -S10% -k1rn > entityfreq`

# Datasets

By filtering classes with at least a certain number of examples of support, we can change the number of actions in the dataset.  In our datasets we include the same number of examples for every class.

| Dataset | Actions | Examples Per Action | Examples |
| --- | ---- | --- | --- |
| oneshotwiki-311 | 311 | 2000 | 622000 |
| oneshotwiki-14031 | 14031 | 200 | 2806200 |
