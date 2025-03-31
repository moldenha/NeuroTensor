with open('benchmark.txt', 'r') as f:
    lines = f.readlines()


mkl_times = []
nt_times = []
for line in lines:
    mkl_times.append(int(line.rstrip().split(' ')[0]))
    nt_times.append(int(line.rstrip().split(' ')[1]))


mkl_avg = sum(mkl_times) / len(mkl_times)
nt_avg = sum(nt_times) / len(nt_times)
print("mkl average: {}, NeuroTensor average: {}, NeuroTensor is {} times faster than the MKL cblas_sgemm_64 route".format(mkl_avg, nt_avg, mkl_avg / nt_avg))

import matplotlib.pyplot as plt
plt.plot(nt_times, label='NeuroTensor', color='blue')
# plt.plot(nt_times, label='NeuroTensor', color='blue', linewidth=2, linestyle='--', marker='o') # Thicker dashed blue line
plt.plot(mkl_times, label = 'MKL cblas_sgemm_64', color='orange') 
# plt.plot(mkl_times, label = 'MKL cblas_sgemm_64', color='orange', linewidth=2, linestyle='-', marker='s') # Thicker solid orange line


plt.xlabel('Square (Rows = Columns) Matrix Sizes')
plt.ylabel("Microseconds")
plt.title("Macbook Pro 2.3 GHz 8-Core Intel Core i9 Processor")
plt.legend()
plt.show()
