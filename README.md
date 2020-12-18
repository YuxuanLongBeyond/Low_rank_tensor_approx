# Low_rank_tensor_approx


Please ensure that Numpy and Scipy are installed. Python3 is required.
  
To reproduce the results for low-rank solver (ALS), please directly run the script 'main.py'. The results, by default, correspond to one case in the Table 1 in the report, where initial tensor rank p0 = 50 for n = 200. To reproduce other results in Table 1, please change the parameter p0 in the main script. Running once would cost about 20 minutes.  

To test the direct solver (not recommended), please change the line 'use_direct_solve = 0' to 'use_direct_solve = 1' in the main script.  

## Remark
The reported time consumption may not be reproducible. For instance, the results in Figure 1 are not reproducible.
