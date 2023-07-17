EXECUTABLES = conv sum_red vector_add naive_mat_mul cache_tiled_mat_mul coal_mat_mul

all: $(EXECUTABLES)

$(EXECUTABLES): %: %.cu
	nvcc -o $@ $<

clean:
	rm -f $(EXECUTABLES)
