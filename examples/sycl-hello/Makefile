CXX      := icpx
CXXFLAGS := -std=c++17 -O3 -fsycl -fsycl-targets=spir64
TARGET   := vecadd

$(TARGET): vecadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: clean
