// Direct-summation N-body for a two-galaxy collision, SYCL + MPI.
//
// Every rank holds a replicated copy of all positions and computes forces for
// a contiguous slice of particles on its own GPU. After each drift the slices
// are stitched back together with an Allgatherv, so the force evaluation is
// always against the full system. Integration is kick-drift-kick leapfrog with
// Plummer softening, which is time-reversible and conserves energy well at
// fixed step size.
//
// Cost is O(N^2 / nranks) per step. That is the point: no tree, no multipole
// approximation, so the only error terms are the softening and the timestep.

#include <sycl/sycl.hpp>
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 256;

struct Args {
  std::string ic = "ic.bin";
  std::string outdir = "snaps";
  double dt = 0.02;
  int64_t steps = 20000;
  int64_t snap_every = 100;
  float eps = 0.08f;
};

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (k == "--ic") a.ic = next();
    else if (k == "--out") a.outdir = next();
    else if (k == "--dt") a.dt = std::stod(next());
    else if (k == "--steps") a.steps = std::stoll(next());
    else if (k == "--snap-every") a.snap_every = std::stoll(next());
    else if (k == "--eps") a.eps = std::stof(next());
  }
  return a;
}

// a_i = sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^{3/2}
//
// The j == i term vanishes because its numerator is the zero vector, so no
// branch is needed to skip self-interaction.
sycl::event compute_accel(sycl::queue &q, const sycl::float4 *pos,
                          sycl::float4 *acc, int64_t N, int64_t i0,
                          int64_t my_n, float eps2) {
  const int64_t groups = (my_n + kTile - 1) / kTile;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<sycl::float4, 1> sh(sycl::range<1>(kTile), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kTile), sycl::range<1>(kTile)),
        [=](sycl::nd_item<1> it) {
          const int64_t li = it.get_global_id(0);
          const int lid = static_cast<int>(it.get_local_id(0));
          const bool active = li < my_n;

          const sycl::float4 p = active ? pos[i0 + li] : sycl::float4{0, 0, 0, 0};
          float ax = 0.f, ay = 0.f, az = 0.f;

          for (int64_t base = 0; base < N; base += kTile) {
            const int64_t idx = base + lid;
            sh[lid] = (idx < N) ? pos[idx] : sycl::float4{0, 0, 0, 0};
            sycl::group_barrier(it.get_group());

            const int lim = static_cast<int>(sycl::min<int64_t>(kTile, N - base));
            for (int j = 0; j < lim; ++j) {
              const sycl::float4 s = sh[j];
              const float dx = s.x() - p.x();
              const float dy = s.y() - p.y();
              const float dz = s.z() - p.z();
              const float r2 = dx * dx + dy * dy + dz * dz + eps2;
              const float inv = sycl::rsqrt(r2);
              const float f = s.w() * inv * inv * inv;
              ax += f * dx; ay += f * dy; az += f * dz;
            }
            sycl::group_barrier(it.get_group());
          }
          if (active) acc[li] = sycl::float4{ax, ay, az, 0.f};
        });
  });
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  const Args args = parse(argc, argv);

  // ---- read initial conditions (every rank reads; the file is small) ----
  int64_t N = 0;
  std::vector<sycl::float4> h_pos;
  std::vector<float> h_vel;
  {
    FILE *f = std::fopen(args.ic.c_str(), "rb");
    if (!f) { if (!rank) std::fprintf(stderr, "cannot open %s\n", args.ic.c_str()); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (std::fread(&N, sizeof(int64_t), 1, f) != 1) MPI_Abort(MPI_COMM_WORLD, 1);
    std::vector<float> p(3 * N), v(3 * N), m(N);
    std::fread(p.data(), sizeof(float), 3 * N, f);
    std::fread(v.data(), sizeof(float), 3 * N, f);
    std::fread(m.data(), sizeof(float), N, f);
    std::fclose(f);
    h_pos.resize(N);
    for (int64_t i = 0; i < N; ++i)
      h_pos[i] = sycl::float4{p[3*i], p[3*i+1], p[3*i+2], m[i]};
    h_vel.swap(v);
  }

  // ---- contiguous slice per rank ----
  const int64_t base = N / nranks, rem = N % nranks;
  std::vector<int> counts(nranks), displs(nranks);
  int64_t off = 0;
  for (int r = 0; r < nranks; ++r) {
    const int64_t c = base + (r < rem ? 1 : 0);
    counts[r] = static_cast<int>(c * 4);        // in floats
    displs[r] = static_cast<int>(off * 4);
    off += c;
  }
  const int64_t my_n = counts[rank] / 4;
  const int64_t i0 = displs[rank] / 4;

  // ---- one GPU per rank ----
  auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (gpus.empty()) { std::fprintf(stderr, "rank %d: no GPU\n", rank); MPI_Abort(MPI_COMM_WORLD, 2); }
  sycl::queue q{gpus[rank % gpus.size()], sycl::property::queue::in_order{}};

  if (!rank) {
    std::printf("N=%lld  ranks=%d  gpus/node=%zu  dt=%g  steps=%lld  eps=%g\n",
                (long long)N, nranks, gpus.size(), args.dt, (long long)args.steps, args.eps);
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fflush(stdout);
  }

  sycl::float4 *d_pos = sycl::malloc_device<sycl::float4>(N, q);
  sycl::float4 *d_acc = sycl::malloc_device<sycl::float4>(my_n, q);
  std::vector<sycl::float4> h_acc(my_n);
  q.memcpy(d_pos, h_pos.data(), N * sizeof(sycl::float4)).wait();

  const float eps2 = args.eps * args.eps;
  const float dt = static_cast<float>(args.dt);

  // Priming force evaluation so the loop below is a clean one-eval-per-step.
  compute_accel(q, d_pos, d_acc, N, i0, my_n, eps2).wait();
  q.memcpy(h_acc.data(), d_acc, my_n * sizeof(sycl::float4)).wait();

  int64_t snap = 0;
  const auto t_start = std::chrono::steady_clock::now();

  for (int64_t step = 0; step <= args.steps; ++step) {
    if (step % args.snap_every == 0) {
      if (!rank) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/snap_%05lld.bin", args.outdir.c_str(), (long long)snap);
        FILE *f = std::fopen(path, "wb");
        if (f) {
          std::vector<float> xyz(3 * N);
          for (int64_t i = 0; i < N; ++i) {
            xyz[3*i] = h_pos[i].x(); xyz[3*i+1] = h_pos[i].y(); xyz[3*i+2] = h_pos[i].z();
          }
          std::fwrite(xyz.data(), sizeof(float), 3 * N, f);
          std::fclose(f);
        }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::printf("snap %05lld  step %lld/%lld  t=%.2f  elapsed=%.1fs\n",
                    (long long)snap, (long long)step, (long long)args.steps,
                    step * args.dt, el);
        std::fflush(stdout);
      }
      ++snap;
    }
    if (step == args.steps) break;

    // kick(dt/2) -> drift(dt) -> sync -> accel -> kick(dt/2)
    for (int64_t i = 0; i < my_n; ++i) {
      const int64_t g = i0 + i;
      h_vel[3*g+0] += 0.5f * dt * h_acc[i].x();
      h_vel[3*g+1] += 0.5f * dt * h_acc[i].y();
      h_vel[3*g+2] += 0.5f * dt * h_acc[i].z();
      h_pos[g].x() += dt * h_vel[3*g+0];
      h_pos[g].y() += dt * h_vel[3*g+1];
      h_pos[g].z() += dt * h_vel[3*g+2];
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   reinterpret_cast<float *>(h_pos.data()),
                   counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    q.memcpy(d_pos, h_pos.data(), N * sizeof(sycl::float4)).wait();
    compute_accel(q, d_pos, d_acc, N, i0, my_n, eps2).wait();
    q.memcpy(h_acc.data(), d_acc, my_n * sizeof(sycl::float4)).wait();

    for (int64_t i = 0; i < my_n; ++i) {
      const int64_t g = i0 + i;
      h_vel[3*g+0] += 0.5f * dt * h_acc[i].x();
      h_vel[3*g+1] += 0.5f * dt * h_acc[i].y();
      h_vel[3*g+2] += 0.5f * dt * h_acc[i].z();
    }
  }

  if (!rank) {
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    const double pairs = double(N) * double(N) * double(args.steps);
    std::printf("done: %lld steps in %.1f s  (%.2f ms/step, %.3g interactions/s)\n",
                (long long)args.steps, el, 1e3 * el / double(args.steps), pairs / el);
  }

  sycl::free(d_pos, q);
  sycl::free(d_acc, q);
  MPI_Finalize();
  return 0;
}
