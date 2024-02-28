#include "cumc.h"
#include "cudualmc.h"
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace cumc
{

  template <typename Scalar, typename IndexType>
  class CUMC
  {
    CuMC<Scalar, IndexType> mc;

    static_assert(std::is_same<Scalar, double>() ||
                  std::is_same<Scalar, float>());
    static_assert(std::is_same<IndexType, long>() ||
                  std::is_same<IndexType, int>());

  public:
    ~CUMC()
    {
      cudaDeviceSynchronize();
      cudaFree(mc.temp_storage);
      cudaFree(mc.first_cell_used);
      cudaFree(mc.used_to_first_mc_vert);
      cudaFree(mc.used_to_first_mc_tri);
      cudaFree(mc.used_cell_code);
      cudaFree(mc.used_cell_index);
      cudaFree(mc.verts_type);
      cudaFree(mc.tris);
      cudaFree(mc.verts);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid,
                                                     torch::Tensor deform,
                                                     Scalar iso)
    {
      CHECK_INPUT(grid);
      CHECK_INPUT(deform);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(grid.dtype() == scalarType,
                            "grid type must match the mc class");
      TORCH_INTERNAL_ASSERT(deform.dtype() == scalarType,
                            "deformation type must match the mc class");

      torch::ScalarType indexType = torch::kInt;
      if constexpr (std::is_same<IndexType, int>())
      {
        indexType = torch::kInt;
      }
      else
      {
        indexType = torch::kLong;
      }

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      mc.forward(grid.data_ptr<Scalar>(), reinterpret_cast<Vertex<Scalar> *>(deform.data_ptr<Scalar>()), dimX, dimY, dimZ, iso);

      auto verts =
          torch::from_blob(
              mc.verts, torch::IntArrayRef{mc.n_verts, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(scalarType))
              .clone();
      auto tris =
          torch::from_blob(
              mc.tris, torch::IntArrayRef{mc.n_tris, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      return {verts, tris};
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid,
                                                     Scalar iso)
    {
      CHECK_INPUT(grid);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(grid.dtype() == scalarType,
                            "grid type must match the mc class");

      torch::ScalarType indexType = torch::kInt;
      if constexpr (std::is_same<IndexType, int>())
      {
        indexType = torch::kInt;
      }
      else
      {
        indexType = torch::kLong;
      }

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      mc.forward(grid.data_ptr<Scalar>(), NULL, dimX, dimY, dimZ, iso);

      auto verts =
          torch::from_blob(
              mc.verts, torch::IntArrayRef{mc.n_verts, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(scalarType))
              .clone();
      auto tris =
          torch::from_blob(
              mc.tris, torch::IntArrayRef{mc.n_tris, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      return {verts, tris};
    }

    void backward(torch::Tensor grid, torch::Tensor deform, Scalar iso, torch::Tensor adj_verts,
                  torch::Tensor adj_grid, torch::Tensor adj_deform)
    {
      CHECK_INPUT(adj_verts);
      CHECK_INPUT(adj_grid);
      CHECK_INPUT(adj_deform);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(adj_verts.dtype() == scalarType,
                            "adj_verts type must match the mc class");
      TORCH_INTERNAL_ASSERT(adj_grid.dtype() == scalarType,
                            "adj_grid type must match the mc class");
      TORCH_INTERNAL_ASSERT(adj_deform.dtype() == scalarType,
                            "adj_deform type must match the mc class");

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      mc.backward(
          grid.data_ptr<Scalar>(), reinterpret_cast<Vertex<Scalar> *>(deform.data_ptr<Scalar>()), dimX, dimY, dimZ, iso,
          adj_grid.data_ptr<Scalar>(), reinterpret_cast<Vertex<Scalar> *>(adj_deform.data_ptr<Scalar>()),
          reinterpret_cast<Vertex<Scalar> *>(adj_verts.data_ptr<Scalar>()));
    }

    void backward(torch::Tensor grid, Scalar iso, torch::Tensor adj_verts,
                  torch::Tensor adj_grid)
    {
      CHECK_INPUT(adj_verts);
      CHECK_INPUT(adj_grid);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(adj_verts.dtype() == scalarType,
                            "adj_verts type must match the mc class");
      TORCH_INTERNAL_ASSERT(adj_grid.dtype() == scalarType,
                            "adj_grid type must match the mc class");

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      mc.backward(
          grid.data_ptr<Scalar>(), NULL, dimX, dimY, dimZ, iso,
          adj_grid.data_ptr<Scalar>(), NULL,
          reinterpret_cast<Vertex<Scalar> *>(adj_verts.data_ptr<Scalar>()));
    }
  };

} // namespace cumc

namespace cudualmc
{
  template <typename Scalar, typename IndexType>
  class CUDMC
  {
    CUDualMC<Scalar, IndexType> dmc;

    static_assert(std::is_same<Scalar, double>() ||
                  std::is_same<Scalar, float>());
    static_assert(std::is_same<IndexType, long>() ||
                  std::is_same<IndexType, int>());

  public:
    ~CUDMC()
    {
      cudaDeviceSynchronize();
      cudaFree(dmc.temp_storage);
      cudaFree(dmc.first_cell_used);
      cudaFree(dmc.used_to_first_mc_vert);
      cudaFree(dmc.used_to_first_mc_patch);
      cudaFree(dmc.used_cell_code);
      cudaFree(dmc.used_cell_index);
      cudaFree(dmc.mc_vert_to_cell);
      cudaFree(dmc.mc_vert_type);
      cudaFree(dmc.quads);
      cudaFree(dmc.verts);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid,
                                                     torch::Tensor deform,
                                                     Scalar iso)
    {
      CHECK_INPUT(grid);
      CHECK_INPUT(deform);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(grid.dtype() == scalarType,
                            "grid type must match the dmc class");
      TORCH_INTERNAL_ASSERT(deform.dtype() == scalarType,
                            "deformation type must match the dmc class");

      torch::ScalarType indexType = torch::kInt;
      if constexpr (std::is_same<IndexType, int>())
      {
        indexType = torch::kInt;
      }
      else
      {
        indexType = torch::kLong;
      }

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      dmc.forward(grid.data_ptr<Scalar>(), reinterpret_cast<Vertex<Scalar> *>(deform.data_ptr<Scalar>()), dimX, dimY, dimZ, iso);

      auto verts =
          torch::from_blob(
              dmc.verts, torch::IntArrayRef{dmc.n_verts, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(scalarType))
              .clone();
      auto quads =
          torch::from_blob(
              dmc.quads, torch::IntArrayRef{dmc.n_quads, 4},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      return {verts, quads};
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid,
                                                     Scalar iso)
    {
      CHECK_INPUT(grid);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(grid.dtype() == scalarType,
                            "grid type must match the dmc class");

      torch::ScalarType indexType = torch::kInt;
      if constexpr (std::is_same<IndexType, int>())
      {
        indexType = torch::kInt;
      }
      else
      {
        indexType = torch::kLong;
      }

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      dmc.forward(grid.data_ptr<Scalar>(), NULL, dimX, dimY, dimZ, iso);

      auto verts =
          torch::from_blob(
              dmc.verts, torch::IntArrayRef{dmc.n_verts, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(scalarType))
              .clone();
      auto quads =
          torch::from_blob(
              dmc.quads, torch::IntArrayRef{dmc.n_quads, 4},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      return {verts, quads};
    }

    void backward(torch::Tensor grid, torch::Tensor deform, Scalar iso, torch::Tensor adj_verts,
                  torch::Tensor adj_grid, torch::Tensor adj_deform)
    {
      CHECK_INPUT(adj_verts);
      CHECK_INPUT(adj_grid);
      CHECK_INPUT(adj_deform);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(adj_verts.dtype() == scalarType,
                            "adj_verts type must match the dmc class");
      TORCH_INTERNAL_ASSERT(adj_grid.dtype() == scalarType,
                            "adj_grid type must match the dmc class");
      TORCH_INTERNAL_ASSERT(adj_deform.dtype() == scalarType,
                            "adj_deform type must match the dmc class");

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      dmc.backward(
          grid.data_ptr<Scalar>(), reinterpret_cast<Vertex<Scalar> *>(deform.data_ptr<Scalar>()), dimX, dimY, dimZ, iso,
          adj_grid.data_ptr<Scalar>(), reinterpret_cast<Vertex<Scalar> *>(adj_deform.data_ptr<Scalar>()),
          reinterpret_cast<Vertex<Scalar> *>(adj_verts.data_ptr<Scalar>()));
    }

    void backward(torch::Tensor grid, Scalar iso, torch::Tensor adj_verts,
                  torch::Tensor adj_grid)
    {
      CHECK_INPUT(adj_verts);
      CHECK_INPUT(adj_grid);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(adj_verts.dtype() == scalarType,
                            "adj_verts type must match the dmc class");
      TORCH_INTERNAL_ASSERT(adj_grid.dtype() == scalarType,
                            "adj_grid type must match the dmc class");

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      dmc.backward(
          grid.data_ptr<Scalar>(), NULL, dimX, dimY, dimZ, iso,
          adj_grid.data_ptr<Scalar>(), NULL,
          reinterpret_cast<Vertex<Scalar> *>(adj_verts.data_ptr<Scalar>()));
    }
  };

} // namespace cudualmc

template <class C>
void register_mc_class(pybind11::module m, std::string name)
{
  pybind11::class_<C>(m, name)
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, C>(&C::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, C, torch::Tensor, torch::Tensor, torch::Tensor, C>(&C::backward))
      .def("forward", pybind11::overload_cast<torch::Tensor, C>(&C::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, C, torch::Tensor, torch::Tensor, C>(&C::backward));
}

template <class C>
void register_dualmc_class(pybind11::module m, std::string name)
{
  pybind11::class_<C>(m, name)
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, C>(&C::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, C, torch::Tensor, torch::Tensor, torch::Tensor, C>(&C::backward))
      .def("forward", pybind11::overload_cast<torch::Tensor, C>(&C::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, C, torch::Tensor, torch::Tensor, C>(&C::backward));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  pybind11::class_<cumc::CUMC<double, int>>(m, "CUMCDouble")
      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, double>(&cumc::CUMC<double, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor>(&cumc::CUMC<double, int>::backward))
      .def("forward", pybind11::overload_cast<torch::Tensor, double>(&cumc::CUMC<double, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, double, torch::Tensor, torch::Tensor>(&cumc::CUMC<double, int>::backward));

  pybind11::class_<cumc::CUMC<float, int>>(m, "CUMCFloat")
      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, float>(&cumc::CUMC<float, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, float, torch::Tensor, torch::Tensor, torch::Tensor>(&cumc::CUMC<float, int>::backward))
      .def("forward", pybind11::overload_cast<torch::Tensor, float>(&cumc::CUMC<float, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, float, torch::Tensor, torch::Tensor>(&cumc::CUMC<float, int>::backward));
  pybind11::class_<cudualmc::CUDMC<double, int>>(m, "CUDMCDouble")

      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, double>(&cudualmc::CUDMC<double, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor>(&cudualmc::CUDMC<double, int>::backward))
      .def("forward", pybind11::overload_cast<torch::Tensor, double>(&cudualmc::CUDMC<double, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, double, torch::Tensor, torch::Tensor>(&cudualmc::CUDMC<double, int>::backward));

  pybind11::class_<cudualmc::CUDMC<float, int>>(m, "CUDMCFloat")
      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, float>(&cudualmc::CUDMC<float, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, float, torch::Tensor, torch::Tensor, torch::Tensor>(&cudualmc::CUDMC<float, int>::backward))
      .def("forward", pybind11::overload_cast<torch::Tensor, float>(&cudualmc::CUDMC<float, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, float, torch::Tensor, torch::Tensor>(&cudualmc::CUDMC<float, int>::backward));
}