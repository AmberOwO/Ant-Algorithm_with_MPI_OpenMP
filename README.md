# Ant-Algorithm_with_MPI_OpenMP
## 編譯&執行
```
mpic++ -fopenmp Ant-Algorithm.cpp -o Ant-Algorithm
mpiexec -f (host_file) -n (process_num) ./Ant-Algorithm ./(test_file_nuame) (alpha) (beta) (rho)
```
