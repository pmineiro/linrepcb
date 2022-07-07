# Contents

* 1 answer result
  * [Interactive notebook](spannerepsgreedydynamic.ipynb)
    * Exhibits the code to embed the items and predict the tags.
    * If you use a small embedding dimension and are patient you can run this interactively.
  * [Reproduce the paper results](run-spannerepsgreedydynamic.py)
    * Warning: this was done in parallel on a GPU cluster and the results bootstrapped to get confidence intervals.
    * The hyperparameters were selected by [random search](tune-spannerepsgreedydynamic.py)
      * Warning: This was also done in parallel on a GPU cluster.  
* 5 answer result, 3 exploration slots
  * [Interactive notebook](spannerepsgreedydynamic5by3.ipynb)
  * [Reproduce the paper results](run-spannerepsgreedydynamic5by3.py)
    * We were running out of time so we just reused the hyperparameters found by random search from the 1 answer result setting.  If anything, these are sub-optimal, but are still really good.

# How to get the data

* download (using w3m) from https://drive.google.com/u/0/uc?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN&export=download
