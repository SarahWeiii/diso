#include <cstdint>
#include <cuda_runtime.h>

namespace cudualmc
{

  template <typename T>
  struct Vertex
  {
    T x, y, z;

    inline __device__ __host__ T *data_ptr() { return &x; }

    inline __device__ __host__ Vertex<T> operator+(Vertex<T> const &other) const
    {
      return {x + other.x, y + other.y, z + other.z};
    }
    inline __device__ __host__ T dot(Vertex<T> const &other) const
    {
      return x * other.x + y * other.y + z * other.z;
    }

    inline __device__ __host__ Vertex<T> operator-(Vertex<T> const &other) const
    {
      return {x - other.x, y - other.y, z - other.z};
    }

    inline __device__ __host__ Vertex<T> operator*(Vertex<T> const &other) const
    {
      return {x * other.x, y * other.y, z * other.z};
    }

    inline __device__ __host__ Vertex<T> operator*(T const &scalar) const
    {
      return {x * scalar, y * scalar, z * scalar};
    }

    inline __device__ __host__ Vertex<T> &operator+=(Vertex<T> const &other)
    {
      x += other.x;
      y += other.y;
      z += other.z;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator-=(Vertex<T> const &other)
    {
      x -= other.x;
      y -= other.y;
      z -= other.z;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator*=(T const &scalar)
    {
      x *= scalar;
      y *= scalar;
      z *= scalar;
      return *this;
    }
  };

  template <typename T>
  struct Quad
  {
    T i, j, k, l;
    inline __device__ __host__ T *data_ptr() { return &i; }
  };

  template <typename Scalar, typename IndexType>
  struct CUDualMC
  {
    IndexType dims[3]{};
    IndexType n_cells{};

    IndexType n_used_cells{0};
    IndexType n_verts{0};
    IndexType n_quads{0};

    // temp storage
    size_t allocated_temp_storage_size{};
    IndexType *__restrict__ temp_storage{}; // used for prefix sum

    size_t allocated_cell_count{};
    IndexType *__restrict__ first_cell_used{}; // cell to used cell index

    //  used cell
    size_t allocated_used_cell_count{};
    IndexType *__restrict__ used_cell_index{};        // used cell to cell index
    IndexType *__restrict__ used_to_first_mc_vert{};  // used cell to mc vertex index
    uint8_t *__restrict__ used_cell_code{};           // used cell to cube code
    IndexType *__restrict__ used_to_first_mc_patch{}; // used cell to mc patch index

    // TODO
    // IndexType *__restrict__ used_cell_mc_vert{};

    // quads
    size_t allocated_quad_count{};
    IndexType *__restrict__ mc_vert_to_cell{}; // vert index to cell index
    // 0: x entering, 1: y entering, 2: z entering
    // 3: x exiting, 4: y exiting, 5: z exiting
    uint8_t *__restrict__ mc_vert_type{};
    Quad<IndexType> *__restrict__ quads{}; // output quads

    // TODO
    // Vertex<Scalar> *__restrict__ mc_verts{};

    // output
    size_t allocated_vert_count{};
    Vertex<Scalar> *__restrict__ verts{}; // output verts

    inline __device__ __host__ IndexType gA(IndexType const x, IndexType const y,
                                            IndexType const z) const
    {
      return z + dims[2] * (y + dims[1] * x);
    }
    inline __device__ __host__ IndexType gX(IndexType const linearizedCellID) const
    {
      return linearizedCellID / (dims[2] * dims[1]);
    }
    inline __device__ __host__ IndexType gY(IndexType const linearizedCellID) const
    {
      return (linearizedCellID / dims[2]) % dims[1];
    }
    inline __device__ __host__ IndexType gZ(IndexType const linearizedCellID) const
    {
      return linearizedCellID % dims[2];
    }

    inline __host__ void resize(IndexType x, IndexType y, IndexType z)
    {
      dims[0] = x;
      dims[1] = y;
      dims[2] = z;
      n_cells = x * y * z;
    }

    __host__ void ensure_temp_storage_size(size_t size);
    __host__ void ensure_cell_storage_size(size_t cell_count);
    __host__ void ensure_used_cell_storage_size(size_t cell_count);
    __host__ void ensure_quad_storage_size(size_t quad_count);
    __host__ void ensure_vert_storage_size(size_t vert_count);
    __host__ void ensure_edge_storage_size(size_t edge_count);

    __host__ void forward(Scalar const *d_data, Vertex<Scalar> const *d_deform, IndexType dimX, IndexType dimY, IndexType dimZ,
                          Scalar iso);
    __host__ void backward(Scalar const *d_data, Vertex<Scalar> const *d_deform, IndexType dimX, IndexType dimY,
                           IndexType dimZ, Scalar iso, Scalar *adj_d_data, Vertex<Scalar> *adj_d_deform,
                           Vertex<Scalar> const *adj_verts);
  };

} // namespace cudualmc
