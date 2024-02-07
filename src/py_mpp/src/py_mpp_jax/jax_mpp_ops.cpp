#include <pybind11/pybind11.h>
#include <iostream>
#include <cstring>
#include <type_traits>

extern "C" void py_mpp_define_layout2D(int*, int*, int*, int*, int*);
extern "C" void py_mpp_update_domain2D_r4_3dv(float*, int*, int*, int*, float*, int*, int*, int*, int*, int*);
extern "C" void py_mpp_update_domain2D_r8_3dv(double*, int*, int*, int*, double*, int*, int*, int*, int*, int*);

namespace {

template<typename FP>
void py_mpp_update_domain2D_3dv(FP* fieldx, int* xi, int* xj, int* xk,
                                FP* fieldy, int* yi, int* yj, int* yk,
                                int* domain_handle, int* grid_type);

template<>
void py_mpp_update_domain2D_3dv(float* fieldx, int* xi, int* xj, int* xk,
                                float* fieldy, int* yi, int* yj, int* yk,
                                int* domain_handle, int* grid_type) {
  py_mpp_update_domain2D_r4_3dv(fieldx, xi, xj, xk, fieldy, yi, yj, yk, domain_handle, grid_type);
}

template<>
void py_mpp_update_domain2D_3dv(double* fieldx, int* xi, int* xj, int* xk,
                                double* fieldy, int* yi, int* yj, int* yk,
                                int* domain_handle, int* grid_type) {
  py_mpp_update_domain2D_r8_3dv(fieldx, xi, xj, xk, fieldy, yi, yj, yk, domain_handle, grid_type);
}

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}


template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}


void jax_py_mpp_define_layout2D(void* out_tuple, const void** in) {
    // Parse the inputs
    int* global_indices = (int*)in[0];
    int* n_indices = (int*)in[1];
    int* ndivs = (int*)in[2];
    int* layout = (int*)in[3];
    int* n_layout = (int*)in[4];
    
    std::cout << "input: global_indices = [";
    std::cout << global_indices[0] << ",";
    std::cout << global_indices[1] << ",";
    std::cout << global_indices[2] << ",";
    std::cout << global_indices[3];
    std::cout << "]\n";
    std::cout << "input: layout = [";
    std::cout << layout[0] << ",";
    std::cout << layout[1];
    std::cout << "]\n";
    // The output is stored as a list of pointers since we have multiple outputs
    void **out = reinterpret_cast<void **>(out_tuple);
    int* out_global_indices = (int*)(out[0]);
    int* out_layout = (int*)(out[1]);

    py_mpp_define_layout2D(global_indices, n_indices, ndivs, out_layout, n_layout);
}


template<typename FP>
void jax_py_mpp_update_domain2d_3dv(void *out_tuple, const void **in) {
    
    // Parse the inputs
    FP* fieldx = (FP*)in[0];
    int* xi = (int*)in[1];
    int* xj = (int*)in[2];
    int* xk = (int*)in[3];
    FP* fieldy = (FP*)in[4];
    int* yi = (int*)in[5];
    int* yj = (int*)in[6];
    int* yk = (int*)in[7];
    int* domain_handle = (int*)in[8];
    int* grid_type = (int*)in[9];

      // The output is stored as a list of pointers since we have multiple outputs
    void **out = reinterpret_cast<void **>(out_tuple);
    FP* out_fieldx = (FP*)(out[0]);
    FP* out_fieldy = (FP*)(out[1]);

    std::memcpy((void*)out_fieldx, (const void*)fieldx, (*xi) * (*xj) * (*xk) * sizeof(FP));
    std::memcpy((void*)out_fieldy, (const void*)fieldy, (*yi) * (*yj) * (*yk) * sizeof(FP));

    py_mpp_update_domain2D_3dv(out_fieldx, xi, xj, xk, out_fieldy, yi, yj, yk, domain_handle, grid_type);
}


pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["jax_py_mpp_define_layout2D"] = EncapsulateFunction(jax_py_mpp_define_layout2D);
  dict["jax_py_mpp_update_domain2d_r4_3dv"] = EncapsulateFunction(jax_py_mpp_update_domain2d_3dv<float>);
  dict["jax_py_mpp_update_domain2d_r8_3dv"] = EncapsulateFunction(jax_py_mpp_update_domain2d_3dv<double>);
  return dict;
}

PYBIND11_MODULE(jax_mpp_ops, m) { m.def("registrations", &Registrations); }

}  // namespace