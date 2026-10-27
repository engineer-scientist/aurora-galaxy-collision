// SAXPY on one Aurora GPU tile, using SYCL USM device allocations.
//
//   y = alpha * x + y
//
// Also enumerates every SYCL device the runtime exposes, which on an Aurora
// compute node is 6 Max 1550 GPUs. Under the default FLAT device hierarchy
// each of the 2 tiles per GPU is reported as its own device, so expect 12.

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kN = 1 << 24;  // 16.7M floats, 64 MiB per array
constexpr float kAlpha = 2.5f;

void list_devices() {
  std::cout << "Visible SYCL devices\n";
  for (const auto &platform : sycl::platform::get_platforms()) {
    const auto devices = platform.get_devices();
    if (devices.empty()) continue;

    std::cout << "  " << platform.get_info<sycl::info::platform::name>() << "\n";
    for (const auto &d : devices) {
      std::cout << "    - " << d.get_info<sycl::info::device::name>() << "  ["
                << d.get_info<sycl::info::device::max_compute_units>() << " CUs, "
                << (d.get_info<sycl::info::device::global_mem_size>() >> 30)
                << " GiB]\n";
    }
  }
  std::cout << "\n";
}

}  // namespace

int main() {
  list_devices();

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  std::cout << "Running on: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

  std::vector<float> x(kN, 1.0f);
  std::vector<float> y(kN, 2.0f);

  float *dx = sycl::malloc_device<float>(kN, q);
  float *dy = sycl::malloc_device<float>(kN, q);

  q.memcpy(dx, x.data(), kN * sizeof(float));
  q.memcpy(dy, y.data(), kN * sizeof(float));
  q.wait();

  // Warm up once so the timed launch does not pay JIT or first-touch costs.
  q.parallel_for(sycl::range<1>{kN},
                 [=](sycl::id<1> i) { dy[i] = kAlpha * dx[i] + dy[i]; })
      .wait();
  q.memcpy(dy, y.data(), kN * sizeof(float)).wait();

  const auto t0 = std::chrono::steady_clock::now();
  q.parallel_for(sycl::range<1>{kN},
                 [=](sycl::id<1> i) { dy[i] = kAlpha * dx[i] + dy[i]; })
      .wait();
  const auto t1 = std::chrono::steady_clock::now();

  q.memcpy(y.data(), dy, kN * sizeof(float)).wait();

  const float expected = kAlpha * 1.0f + 2.0f;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < kN; ++i) {
    if (std::fabs(y[i] - expected) > 1e-5f) ++bad;
  }

  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const double gbs = 3.0 * kN * sizeof(float) / secs / 1e9;  // 2 reads + 1 write

  std::cout << "SAXPY  n=" << kN << "  time=" << secs * 1e3 << " ms  " << gbs
            << " GB/s\n"
            << (bad == 0 ? "PASS" : "FAIL") << "  (" << bad << " mismatches)\n";

  sycl::free(dx, q);
  sycl::free(dy, q);
  return bad == 0 ? 0 : 1;
}
