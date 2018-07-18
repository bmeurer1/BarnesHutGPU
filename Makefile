all: barnesHut

barnesHut: bh.cu
	nvcc -o bh bh.cu
