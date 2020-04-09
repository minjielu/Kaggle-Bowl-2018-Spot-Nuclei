# Kaggle-Bowl-2018-Spot-Nuclei
This repository contains codes and results for nuclei recognition challenge of Kaggle

Link to the competition: https://www.kaggle.com/c/data-science-bowl-2018

Link to the Kaggle notebook: https://www.kaggle.com/minjielu/spot-nuclei-traditional-solution
  
   At the beginning of the competition, I don't know about mask-rcnn. This traditional solution still gives me a score of 0.188 and top 19% rank.
   Given that images come in various forms, it definitely can't even see the neck of the score of 0.631 of the best Mask-rcnn model. After the
   competition, I learnt about mask-rcnn and built a picture of the whole frame in my head through literatures.
   
   Contents:
   1. 'spot_nuclei.ipynb' contains illustration of procedures of my clumsy method.
   2. 'spot_nuclei.py' processes all test files and generates results according to the requirement of the competition.
   3. Test folder is divided by me to 11 parts to be parallelly processed by several machines. Some partial results are in 'Partial_results'.
      'combine_result.py' combines these results to a single file.
   
