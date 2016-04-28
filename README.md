# Xtracta_supplier
The file " supplier_detect_code" requires only two inputs: 
1. The name of the invoice text file generated by the OCR engine (Line 97)
2. The name of supplier_names text file (line 96)


#Main advantages of the algorithm: 
1. It is efficient because instead of using complete lists, we use sets. For example, the word 'of' is repeated 4 times in the supplier names list. If we compare any invoice word, we should not be comparing it to 'of' 4 times. The use of set removes any repeated elements from the list. 
2. The algorithm can handle very large data. We use sparse matrices which removes any problems faced in storage of big data. 
3. The algorithm is based on probability density and joint probability maps.  

#Future possible additions: 
1. Use of string distance metrics 
2. Parallelization to improve speed 
