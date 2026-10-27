MPICXX   := mpicxx
CXXFLAGS := -std=c++17 -O3 -fsycl -fsycl-targets=spir64 -ffast-math
TARGET   := nbody

$(TARGET): nbody.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: clean
