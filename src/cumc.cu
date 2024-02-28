#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cub/device/device_scan.cuh>

#include "cumc.h"

//  Coordinate system
//
//       y
//       |
//       |
//       |
//       0-----x
//      /
//     /
//    z
//

// Cell Corners
// (Corners are voxels. Number correspond to Morton codes of corner coordinates)
//
//       2-------------------3
//      /|                  /|
//     / |                 / |
//    /  |                /  |
//   6-------------------7   |
//   |   |               |   |
//   |   |               |   |
//   |   |               |   |
//   |   |               |   |
//   |   0---------------|---1
//   |  /                |  /
//   | /                 | /
//   |/                  |/
//   4-------------------5
//

//         Cell Edges
//
//       o--------4----------o
//      /|                  /|
//     7 |                 5 |
//    /  |                /  |
//   o--------6----------o   |
//   |   8               |   9
//   |   |               |   |
//   |   |               |   |
//   11  |               10  |
//   |   o--------0------|---o
//   |  /                |  /
//   | 3                 | 1
//   |/                  |/
//   o--------2----------o
//

// Encodes the edge vertices for the 256 marching cubes cases.
// A marching cube case produces up to four faces and ,thus, up to four
// dual points.

namespace cumc
{

  bool check_cuda_result(cudaError_t code, const char *file, int line)
  {
    if (code == cudaSuccess)
      return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file,
            line);
    return false;
  }

#define CHECK_CUDA(code) check_cuda_result(code, __FILE__, __LINE__);

  template <typename T>
  inline __device__ __host__ T min(T a, T b)
  {
    return a < b ? a : b;
  }
  template <typename T>
  inline __device__ __host__ T max(T a, T b) { return a > b ? a : b; }
  template <typename T>
  inline __device__ __host__ T clamp(T x, T a, T b)
  {
    return min(max(a, x), b);
  }

  constexpr int BLOCK_SIZE = 512;

  // __constant__ int mcCorners[8][3] = {
  //     {0, 0, 0},
  //     {1, 0, 0},
  //     {0, 1, 0},
  //     {1, 1, 0},
  //     {0, 0, 1},
  //     {1, 0, 1},
  //     {0, 1, 1},
  //     {1, 1, 1},
  // };

  // warp
  __constant__ int mcCorners[8][3] = { {0,0,0}, {1,0,0},{1,1,0},{0,1,0}, {0,0,1}, {1,0,1},{1,1,1},{0,1,1} };

  __constant__ int mcEdgeLocations[12][4] = {
      {0, 0, 0, 0},
      {1, 0, 0, 2},
      {0, 0, 1, 0},
      {0, 0, 0, 2},
      {0, 1, 0, 0},
      {1, 1, 0, 2},
      {0, 1, 1, 0},
      {0, 1, 0, 2},
      {0, 0, 0, 1},
      {1, 0, 0, 1},
      {1, 0, 1, 1},
      {0, 0, 1, 1},
  };

  // // warp
  // __constant__ int mcEdgeLocations[12][4] = {
  //       // relative cell coords, edge within cell
  //       {0, 0, 0,  0},
  //       {1, 0, 0,  1},
  //       {0, 1, 0,  0},
  //       {0, 0, 0,  1},

  //       {0, 0, 1,  0},
  //       {1, 0, 1,  1},
  //       {0, 1, 1,  0},
  //       {0, 0, 1,  1},

  //       {0, 0, 0,  2},
  //       {1, 0, 0,  2},
  //       {1, 1, 0,  2},
  //       {0, 1, 0,  2}
  //   };

  __constant__ int firstMarchingCubesId[257] = {
      0, 0, 3, 6, 12, 15, 21, 27, 36, 39, 45, 51, 60, 66, 75, 84, 90, 93, 99, 105, 114,
      120, 129, 138, 150, 156, 165, 174, 186, 195, 207, 219, 228, 231, 237, 243, 252, 258, 267, 276, 288,
      294, 303, 312, 324, 333, 345, 357, 366, 372, 381, 390, 396, 405, 417, 429, 438, 447, 459, 471, 480,
      492, 507, 522, 528, 531, 537, 543, 552, 558, 567, 576, 588, 594, 603, 612, 624, 633, 645, 657, 666,
      672, 681, 690, 702, 711, 723, 735, 750, 759, 771, 783, 798, 810, 825, 840, 852, 858, 867, 876, 888,
      897, 909, 915, 924, 933, 945, 957, 972, 984, 999, 1008, 1014, 1023, 1035, 1047, 1056, 1068, 1083, 1092, 1098,
      1110, 1125, 1140, 1152, 1167, 1173, 1185, 1188, 1191, 1197, 1203, 1212, 1218, 1227, 1236, 1248, 1254, 1263, 1272, 1284,
      1293, 1305, 1317, 1326, 1332, 1341, 1350, 1362, 1371, 1383, 1395, 1410, 1419, 1425, 1437, 1446, 1458, 1467, 1482, 1488,
      1494, 1503, 1512, 1524, 1533, 1545, 1557, 1572, 1581, 1593, 1605, 1620, 1632, 1647, 1662, 1674, 1683, 1695, 1707, 1716,
      1728, 1743, 1758, 1770, 1782, 1791, 1806, 1812, 1827, 1839, 1845, 1848, 1854, 1863, 1872, 1884, 1893, 1905, 1917, 1932,
      1941, 1953, 1965, 1980, 1986, 1995, 2004, 2010, 2019, 2031, 2043, 2058, 2070, 2085, 2100, 2106, 2118, 2127, 2142, 2154,
      2163, 2169, 2181, 2184, 2193, 2205, 2217, 2232, 2244, 2259, 2268, 2280, 2292, 2307, 2322, 2328, 2337, 2349, 2355, 2358,
      2364, 2373, 2382, 2388, 2397, 2409, 2415, 2418, 2427, 2433, 2445, 2448, 2454, 2457, 2460, 2460};

  __constant__ int marchingCubesIds[2460] = {
    0, 3, 8, 0, 9, 1, 9, 3, 8, 1, 3, 9, 9, 4, 5, 0, 3, 8, 9, 4, 5, 1, 4, 5, 0, 4, 1, 4, 3, 8, 4, 5, 3, 5, 1, 3, 8, 7, 4, 0, 7, 4, 3, 7, 0, 9, 1, 0, 4, 8, 7, 9, 7, 4, 9, 1, 7, 1, 3, 7, 8, 5, 9, 7, 5, 8, 0, 5, 9, 0, 3, 5, 3, 7, 5, 8, 1, 0, 8, 7, 1, 7, 5, 1, 1, 3, 5, 5, 3, 7, 2, 11, 3, 2, 8, 0, 11, 8, 2, 0, 9, 1, 3, 2, 11, 2, 9, 1, 2, 11, 9, 11, 8, 9, 9, 4, 5, 3, 2, 11, 8, 2, 11, 8, 0, 2, 9, 4, 5, 1, 4, 5, 1, 0, 4, 3, 2, 11, 4, 5, 1, 4, 1, 11, 4, 11, 8, 11, 1, 2, 3, 2, 11, 8, 7, 4, 7, 2, 11, 7, 4, 2, 4, 0, 2, 1, 0, 9, 3, 2, 11, 4, 8, 7, 2, 11, 7, 1, 2, 7, 1, 7, 4, 1, 4, 9, 8, 5, 9, 8, 7, 5, 11, 3, 2, 9, 7, 5, 9, 2, 7, 9, 0, 2, 11, 7, 2, 2, 11, 3, 1, 0, 7, 1, 7, 5, 7, 0, 8, 2, 11, 7, 2, 7, 1, 1, 7, 5, 1, 10, 2, 1, 10, 2, 0, 3, 8, 0, 10, 2, 9, 10, 0, 3, 10, 2, 3, 8, 10, 8, 9, 10, 9, 4, 5, 1, 10, 2, 8, 0, 3, 9, 4, 5, 2, 1, 10, 10, 4, 5, 10, 2, 4, 2, 0, 4, 4, 5, 10, 8, 4, 10, 8, 10, 2, 8, 2, 3, 1, 10, 2, 4, 8, 7, 0, 7, 4, 0, 3, 7, 2, 1, 10, 0, 10, 2, 0, 9, 10, 4, 8, 7, 4, 9, 10, 4, 10, 3, 4, 3, 7, 2, 3, 10, 5, 8, 7, 5, 9, 8, 1, 10, 2, 2, 1, 10, 0, 3, 9, 3, 5, 9, 3, 7, 5, 10, 2, 0, 10, 0, 7, 10, 7, 5, 7, 0, 8, 10, 2, 3, 10, 3, 5, 5, 3, 7, 1, 11, 3, 10, 11, 1, 1, 8, 0, 1, 10, 8, 10, 11, 8, 0, 11, 3, 0, 9, 11, 9, 10, 11, 9, 10, 8, 8, 10, 11, 1, 11, 3, 1, 10, 11, 5, 9, 4, 5, 9, 4, 1, 10, 0, 10, 8, 0, 10, 11, 8, 3, 0, 4, 3, 4, 10, 3, 10, 11, 5, 10, 4, 4, 5, 10, 4, 10, 8, 8, 10, 11, 11, 1, 10, 11, 3, 1, 8, 7, 4, 1, 10, 11, 1, 11, 4, 1, 4, 0, 4, 11, 7, 4, 8, 7, 0, 9, 3, 9, 11, 3, 9, 10, 11, 7, 4, 9, 7, 9, 11, 11, 9, 10, 1, 10, 3, 3, 10, 11, 5, 9, 8, 5, 8, 7, 10, 11, 0, 10, 0, 1, 11, 7, 0, 9, 0, 5, 7, 5, 0, 7, 5, 0, 7, 0, 8, 5, 10, 0, 3, 0, 11, 10, 11, 0, 7, 5, 10, 11, 7, 10, 5, 6, 10, 0, 3, 8, 10, 5, 6, 1, 0, 9, 10, 5, 6, 9, 3, 8, 9, 1, 3, 10, 5, 6, 9, 6, 10, 4, 6, 9, 9, 6, 10, 9, 4, 6, 8, 0, 3, 1, 6, 10, 1, 0, 6, 0, 4, 6, 10, 1, 3, 10, 3, 4, 10, 4, 6, 8, 4, 3, 4, 8, 7, 5, 6, 10, 7, 0, 3, 7, 4, 0, 5, 6, 10, 0, 9, 1, 4, 8, 7, 10, 5, 6, 10, 5, 6, 9, 1, 4, 1, 7, 4, 1, 3, 7, 6, 8, 7, 6, 10, 8, 10, 9, 8, 0, 3, 7, 0, 7, 10, 0, 10, 9, 10, 7, 6, 8, 7, 6, 0, 8, 6, 0, 6, 10, 0, 10, 1, 6, 10, 1, 6, 1, 7, 7, 1, 3, 10, 5, 6, 2, 11, 3, 2, 8, 0, 2, 11, 8, 6, 10, 5, 9, 1, 0, 10, 5, 6, 3, 2, 11, 5, 6, 10, 9, 1, 11, 9, 11, 8, 11, 1, 2, 6, 9, 4, 6, 10, 9, 2, 11, 3, 9, 4, 10, 10, 4, 6, 8, 0, 2, 8, 2, 11, 3, 2, 11, 1, 0, 10, 0, 6, 10, 0, 4, 6, 11, 8, 1, 11, 1, 2, 8, 4, 1, 10, 1, 6, 4, 6, 1, 8, 7, 4, 11, 3, 2, 5, 6, 10, 10, 5, 6, 2, 11, 4, 2, 4, 0, 4, 11, 7, 0, 9, 1, 2, 11, 3, 4, 8, 7, 10, 5, 6, 1, 4, 9, 1, 7, 4, 1, 2, 7, 11, 7, 2, 10, 5, 6, 3, 2, 11, 8, 7, 10, 8, 10, 9, 10, 7, 6, 10, 9, 7, 10, 7, 6, 9, 0, 7, 11, 7, 2, 0, 2, 7, 0, 10, 1, 0, 6, 10, 0, 8, 6, 7, 6, 8, 3, 2, 11, 6, 10, 1, 6, 1, 7, 2, 11, 1, 11, 7, 1, 5, 2, 1, 6, 2, 5, 2, 5, 6, 2, 1, 5, 0, 3, 8, 5, 0, 9, 5, 6, 0, 6, 2, 0, 3, 8, 9, 3, 9, 6, 3, 6, 2, 6, 9, 5, 9, 2, 1, 9, 4, 2, 4, 6, 2, 8, 0, 3, 9, 4, 1, 4, 2, 1, 4, 6, 2, 0, 4, 2, 2, 4, 6, 3, 8, 4, 3, 4, 2, 2, 4, 6, 5, 2, 1, 5, 6, 2, 7, 4, 8, 0, 3, 4, 4, 3, 7, 2, 1, 5, 2, 5, 6, 8, 7, 4, 0, 9, 6, 0, 6, 2, 6, 9, 5, 6, 2, 9, 6, 9, 5, 2, 3, 9, 4, 9, 7, 3, 7, 9, 1, 6, 2, 1, 8, 6, 1, 9, 8, 7, 6, 8, 3, 7, 9, 3, 9, 0, 7, 6, 9, 1, 9, 2, 6, 2, 9, 8, 7, 6, 8, 6, 0, 0, 6, 2, 6, 2, 3, 7, 6, 3, 11, 5, 6, 11, 3, 5, 3, 1, 5, 0, 11, 8, 0, 5, 11, 0, 1, 5, 6, 11, 5, 5, 6, 11, 9, 5, 11, 9, 11, 3, 9, 3, 0, 5, 6, 11, 5, 11, 9, 9, 11, 8, 9, 4, 6, 9, 6, 3, 9, 3, 1, 3, 6, 11, 4, 6, 1, 4, 1, 9, 6, 11, 1, 0, 1, 8, 11, 8, 1, 11, 3, 0, 11, 0, 6, 6, 0, 4, 11, 8, 4, 6, 11, 4, 4, 8, 7, 5, 6, 3, 5, 3, 1, 3, 6, 11, 4, 0, 11, 4, 11, 7, 0, 1, 11, 6, 11, 5, 1, 5, 11, 9, 3, 0, 9, 11, 3, 9, 5, 11, 6, 11, 5, 4, 8, 7, 7, 4, 9, 7, 9, 11, 5, 6, 9, 6, 11, 9, 3, 1, 6, 3, 6, 11, 1, 9, 6, 7, 6, 8, 9, 8, 6, 0, 1, 9, 7, 6, 11, 11, 3, 0, 11, 0, 6, 8, 7, 0, 7, 6, 0, 11, 7, 6, 11, 6, 7, 8, 0, 3, 7, 11, 6, 0, 9, 1, 7, 11, 6, 3, 9, 1, 3, 8, 9, 7, 11, 6, 5, 9, 4, 6, 7, 11, 9, 4, 5, 8, 0, 3, 6, 7, 11, 4, 1, 0, 4, 5, 1, 6, 7, 11, 6, 7, 11, 4, 5, 8, 5, 3, 8, 5, 1, 3, 11, 4, 8, 6, 4, 11, 11, 0, 3, 11, 6, 0, 6, 4, 0, 4, 11, 6, 4, 8, 11, 0, 9, 1, 9, 6, 4, 9, 3, 6, 9, 1, 3, 3, 11, 6, 5, 11, 6, 5, 9, 11, 9, 8, 11, 5, 11, 6, 9, 11, 5, 9, 3, 11, 9, 0, 3, 0, 8, 11, 0, 11, 5, 0, 5, 1, 6, 5, 11, 11, 6, 5, 11, 5, 3, 3, 5, 1, 6, 3, 2, 7, 3, 6, 8, 6, 7, 8, 0, 6, 0, 2, 6, 3, 6, 7, 3, 2, 6, 1, 0, 9, 1, 2, 6, 1, 6, 8, 1, 8, 9, 7, 8, 6, 6, 3, 2, 6, 7, 3, 4, 5, 9, 9, 4, 5, 8, 0, 7, 0, 6, 7, 0, 2, 6, 2, 7, 3, 2, 6, 7, 0, 4, 1, 4, 5, 1, 5, 1, 8, 5, 8, 4, 1, 2, 8, 7, 8, 6, 2, 6, 8, 3, 4, 8, 3, 2, 4, 2, 6, 4, 0, 2, 4, 2, 6, 4, 9, 1, 0, 4, 8, 2, 4, 2, 6, 2, 8, 3, 9, 1, 2, 9, 2, 4, 4, 2, 6, 3, 9, 8, 3, 6, 9, 3, 2, 6, 6, 5, 9, 5, 9, 0, 5, 0, 6, 6, 0, 2, 2, 6, 8, 2, 8, 3, 6, 5, 8, 0, 8, 1, 5, 1, 8, 5, 1, 2, 6, 5, 2, 2, 1, 10, 11, 6, 7, 0, 3, 8, 2, 1, 10, 7, 11, 6, 10, 0, 9, 10, 2, 0, 11, 6, 7, 7, 11, 6, 3, 8, 2, 8, 10, 2, 8, 9, 10, 1, 10, 2, 5, 9, 4, 11, 6, 7, 6, 7, 11, 9, 4, 5, 0, 3, 8, 2, 1, 10, 11, 6, 7, 10, 2, 5, 2, 4, 5, 2, 0, 4, 8, 2, 3, 8, 10, 2, 8, 4, 10, 5, 10, 4, 7, 11, 6, 11, 4, 8, 11, 6, 4, 10, 2, 1, 1, 10, 2, 0, 3, 6, 0, 6, 4, 6, 3, 11, 8, 6, 4, 8, 11, 6, 9, 10, 0, 10, 2, 0, 6, 4, 3, 6, 3, 11, 4, 9, 3, 2, 3, 10, 9, 10, 3, 1, 10, 2, 5, 9, 6, 9, 11, 6, 9, 8, 11, 9, 6, 5, 9, 11, 6, 9, 0, 11, 3, 11, 0, 1, 10, 2, 2, 0, 5, 2, 5, 10, 0, 8, 5, 6, 5, 11, 8, 11, 5, 11, 6, 5, 11, 5, 3, 10, 2, 5, 2, 3, 5, 6, 1, 10, 6, 7, 1, 7, 3, 1, 8, 6, 7, 0, 6, 8, 0, 10, 6, 0, 1, 10, 0, 7, 3, 0, 10, 7, 0, 9, 10, 10, 6, 7, 6, 7, 8, 6, 8, 10, 10, 8, 9, 9, 4, 5, 1, 10, 7, 1, 7, 3, 7, 10, 6, 0, 7, 8, 0, 6, 7, 0, 1, 6, 10, 6, 1, 9, 4, 5, 7, 3, 10, 7, 10, 6, 3, 0, 10, 5, 10, 4, 0, 4, 10, 6, 7, 8, 6, 8, 10, 4, 5, 8, 5, 10, 8, 10, 3, 1, 10, 4, 3, 10, 6, 4, 8, 3, 4, 1, 10, 6, 1, 6, 0, 0, 6, 4, 9, 10, 3, 9, 3, 0, 10, 6, 3, 8, 3, 4, 6, 4, 3, 9, 10, 6, 4, 9, 6, 9, 8, 6, 9, 6, 5, 8, 3, 6, 10, 6, 1, 3, 1, 6, 5, 9, 0, 5, 0, 6, 1, 10, 0, 10, 6, 0, 0, 8, 3, 10, 6, 5, 5, 10, 6, 7, 10, 5, 11, 10, 7, 7, 10, 5, 7, 11, 10, 3, 8, 0, 10, 7, 11, 10, 5, 7, 9, 1, 0, 5, 11, 10, 5, 7, 11, 1, 3, 9, 3, 8, 9, 7, 9, 4, 7, 11, 9, 11, 10, 9, 0, 3, 8, 9, 4, 11, 9, 11, 10, 11, 4, 7, 1, 11, 10, 1, 4, 11, 1, 0, 4, 4, 7, 11, 11, 10, 4, 11, 4, 7, 10, 1, 4, 8, 4, 3, 1, 3, 4, 4, 10, 5, 4, 8, 10, 8, 11, 10, 3, 4, 0, 3, 10, 4, 3, 11, 10, 5, 4, 10, 1, 0, 9, 10, 5, 8, 10, 8, 11, 8, 5, 4, 1, 3, 4, 1, 4, 9, 3, 11, 4, 5, 4, 10, 11, 10, 4, 9, 8, 10, 8, 11, 10, 0, 3, 11, 0, 11, 9, 9, 11, 10, 1, 0, 8, 1, 8, 10, 10, 8, 11, 1, 3, 11, 10, 1, 11, 10, 3, 2, 10, 5, 3, 5, 7, 3, 10, 0, 2, 10, 7, 0, 10, 5, 7, 7, 8, 0, 0, 9, 1, 3, 2, 5, 3, 5, 7, 5, 2, 10, 5, 7, 2, 5, 2, 10, 7, 8, 2, 1, 2, 9, 8, 9, 2, 4, 10, 9, 4, 3, 10, 4, 7, 3, 2, 10, 3, 0, 2, 7, 0, 7, 8, 2, 10, 7, 4, 7, 9, 10, 9, 7, 0, 4, 10, 0, 10, 1, 4, 7, 10, 2, 10, 3, 7, 3, 10, 1, 2, 10, 4, 7, 8, 4, 10, 5, 8, 10, 4, 8, 2, 10, 8, 3, 2, 10, 5, 4, 10, 4, 2, 2, 4, 0, 8, 5, 4, 8, 10, 5, 8, 3, 10, 2, 10, 3, 0, 9, 1, 10, 5, 4, 10, 4, 2, 9, 1, 4, 1, 2, 4, 3, 2, 10, 3, 10, 8, 8, 10, 9, 0, 2, 10, 9, 0, 10, 3, 2, 10, 3, 10, 8, 1, 0, 10, 0, 8, 10, 1, 2, 10, 2, 7, 11, 2, 1, 7, 1, 5, 7, 0, 3, 8, 2, 1, 11, 1, 7, 11, 1, 5, 7, 9, 5, 7, 9, 7, 2, 9, 2, 0, 11, 2, 7, 8, 9, 2, 8, 2, 3, 9, 5, 2, 11, 2, 7, 5, 7, 2, 2, 7, 11, 1, 7, 2, 1, 4, 7, 1, 9, 4, 1, 11, 2, 1, 7, 11, 1, 9, 7, 4, 7, 9, 0, 3, 8, 7, 11, 2, 7, 2, 4, 4, 2, 0, 7, 11, 2, 7, 2, 4, 3, 8, 2, 8, 4, 2, 4, 1, 5, 4, 11, 1, 4, 8, 11, 11, 2, 1, 1, 5, 11, 1, 11, 2, 5, 4, 11, 3, 11, 0, 4, 0, 11, 8, 11, 5, 8, 5, 4, 11, 2, 5, 9, 5, 0, 2, 0, 5, 9, 5, 4, 3, 11, 2, 2, 1, 9, 2, 9, 11, 11, 9, 8, 2, 1, 9, 2, 9, 11, 0, 3, 9, 3, 11, 9, 2, 0, 8, 11, 2, 8, 2, 3, 11, 1, 5, 3, 5, 7, 3, 8, 0, 1, 8, 1, 7, 7, 1, 5, 0, 9, 5, 0, 5, 3, 3, 5, 7, 8, 9, 5, 7, 8, 5, 9, 4, 7, 9, 7, 1, 1, 7, 3, 8, 0, 1, 8, 1, 7, 9, 4, 1, 4, 7, 1, 0, 4, 7, 3, 0, 7, 8, 4, 7, 4, 8, 3, 4, 3, 5, 5, 3, 1, 1, 5, 4, 0, 1, 4, 4, 8, 3, 4, 3, 5, 0, 9, 3, 9, 5, 3, 9, 5, 4, 9, 8, 3, 1, 9, 3, 0, 1, 9, 0, 8, 3};

  __constant__ uint8_t const problematicConfigs[256] = {
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 1, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 3, 255, 255, 2,
      255, 255, 255, 255, 255, 255, 255, 255, 5, 255, 255, 255, 255, 255, 255, 5, 5, 255, 255,
      255, 255, 255, 255, 4, 255, 255, 255, 3, 3, 1, 1, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 5, 255, 5, 255, 5, 255, 255, 255, 255, 255, 255, 255, 3, 255, 255, 255,
      255, 255, 2, 255, 255, 255, 255, 255, 255, 255, 3, 255, 3, 255, 4, 255, 255, 0, 255,
      0, 255, 255, 255, 255, 255, 255, 255, 255, 1, 255, 255, 255, 0, 255, 255, 255, 255, 255,
      255, 255, 1, 255, 255, 255, 1, 255, 4, 2, 255, 255, 255, 2, 255, 255, 255, 255, 0,
      255, 2, 4, 255, 255, 255, 255, 0, 255, 2, 255, 255, 255, 255, 255, 255, 255, 255, 4,
      255, 255, 4, 255, 255, 255, 255, 255, 255};

  template <typename Scalar, typename IndexType>
  void CuMC<Scalar, IndexType>::ensure_temp_storage_size(size_t size)
  {
    if (size > allocated_temp_storage_size)
    {
      allocated_temp_storage_size = size_t(size + size / 5);
      // fprintf(stderr, "allocated_temp_storage_size %ld\n", allocated_temp_storage_size);
      CHECK_CUDA(cudaFree(temp_storage));
      CHECK_CUDA(cudaMalloc((void **)&temp_storage, allocated_temp_storage_size));
    }
  }

  template <typename Scalar, typename IndexType>
  void CuMC<Scalar, IndexType>::ensure_cell_storage_size(size_t cell_count)
  {
    if (cell_count > allocated_cell_count)
    {
      allocated_cell_count = size_t(cell_count + cell_count / 5);
      // fprintf(stderr, "allocated_cell_count %ld\n", allocated_cell_count);
      CHECK_CUDA(cudaFree(first_cell_used));
      CHECK_CUDA(
          cudaMalloc((void **)&first_cell_used, (allocated_cell_count + 1) * sizeof(IndexType)));
    }
  }

  template <typename Scalar, typename IndexType>
  void CuMC<Scalar, IndexType>::ensure_used_cell_storage_size(size_t cell_count)
  {
    if (cell_count > allocated_used_cell_count)
    {
      allocated_used_cell_count = size_t(cell_count + cell_count / 5);
      // fprintf(stderr, "allocated_used_cell_count %ld\n", allocated_used_cell_count);

      CHECK_CUDA(cudaFree(used_to_first_mc_vert));
      CHECK_CUDA(cudaFree(used_to_first_mc_tri));
      CHECK_CUDA(cudaFree(used_cell_code));
      CHECK_CUDA(cudaFree(used_cell_index));

      // TODO
      // CHECK_CUDA(cudaFree(used_cell_mc_vert));
      // CHECK_CUDA(
      //     cudaMalloc((void **)&used_cell_mc_vert, allocated_used_cell_count * 3 *
      //     sizeof(IndexType)));

      CHECK_CUDA(cudaMalloc((void **)&used_to_first_mc_vert,
                            (allocated_used_cell_count + 1) * sizeof(IndexType)));
      CHECK_CUDA(cudaMalloc((void **)&used_to_first_mc_tri,
                            (allocated_used_cell_count + 1) * sizeof(IndexType)));
      CHECK_CUDA(cudaMalloc((void **)&used_cell_code, allocated_used_cell_count * sizeof(uint8_t)));
      CHECK_CUDA(
          cudaMalloc((void **)&used_cell_index, allocated_used_cell_count * sizeof(IndexType)));
    }
  }

  template <typename Scalar, typename IndexType>
  void CuMC<Scalar, IndexType>::ensure_tri_storage_size(size_t tri_count)
  {
    if (tri_count > allocated_tri_count)
    {
      allocated_tri_count = size_t(tri_count + tri_count / 5);
      // fprintf(stderr, "allocated_tri_count %ld\n", allocated_tri_count);

      CHECK_CUDA(cudaFree(tris));

      // TODO
      // CHECK_CUDA(cudaFree(mc_verts));
      // CHECK_CUDA(cudaMalloc((void **)&mc_verts, allocated_tri_count * sizeof(Vertex<Scalar>)));

      CHECK_CUDA(cudaMalloc((void **)&tris, allocated_tri_count * sizeof(Triangle<IndexType>)));
    }
  }

  template <typename Scalar, typename IndexType>
  void CuMC<Scalar, IndexType>::ensure_vert_storage_size(size_t vert_count)
  {
    if (vert_count > allocated_vert_count)
    {
      allocated_vert_count = size_t(vert_count + vert_count / 5);
      // fprintf(stderr, "allocated_vert_count %ld\n", allocated_vert_count);

      CHECK_CUDA(cudaFree(verts));
      CHECK_CUDA(cudaFree(verts_type));

      CHECK_CUDA(cudaMalloc((void **)&verts, allocated_vert_count * sizeof(Vertex<Scalar>)));
      CHECK_CUDA(cudaMalloc((void **)&verts_type, allocated_vert_count * sizeof(IndexType)));
    }
  }

  template <typename Scalar, typename IndexType>
  __global__ void count_used_cells_kernel(CuMC<Scalar, IndexType> mc,
                                          Scalar const *__restrict__ data, Scalar iso)
  {
    int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= mc.n_cells)
      return;

    IndexType x = mc.gX(cell_index);
    IndexType y = mc.gY(cell_index);
    IndexType z = mc.gZ(cell_index);

    mc.first_cell_used[cell_index] = 0;
    if (x >= mc.dims[0] - 1 || y >= mc.dims[1] - 1 || z >= mc.dims[2] - 1)
      return;

    int code = 0;
    for (int i = 0; i < 8; ++i)
    {
      IndexType cxn = x + mcCorners[i][0];
      IndexType cyn = y + mcCorners[i][1];
      IndexType czn = z + mcCorners[i][2];
      if (data[mc.gA(cxn, cyn, czn)] >= iso)
      {
        code |= (1 << i);
      }
    }

    if (code != 0 && code != 255)
    {
      mc.first_cell_used[cell_index] = 1;
    }
  }

  template <typename Scalar, typename IndexType>
  __global__ void index_used_cells_kernel(CuMC<Scalar, IndexType> mc)
  {
    int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= mc.n_cells)
      return;

    int used_index = mc.first_cell_used[cell_index];
    if (mc.first_cell_used[cell_index + 1] - used_index)
    {
      mc.used_cell_index[used_index] = cell_index;
    }
  }

  template <typename Scalar, typename IndexType>
  __global__ void count_cell_mc_verts_kernel(CuMC<Scalar, IndexType> mc,
                                             Scalar const *__restrict__ data, Scalar iso)
  {

    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= mc.n_used_cells)
      return;
    int cell_index = mc.used_cell_index[used_index];

    IndexType x = mc.gX(cell_index);
    IndexType y = mc.gY(cell_index);
    IndexType z = mc.gZ(cell_index);
    // no need to check range of xyz, already checked when creating used cells

    Scalar d0 = data[cell_index];
    Scalar dx = data[mc.gA(x + 1, y, z)];
    Scalar dy = data[mc.gA(x, y + 1, z)];
    Scalar dz = data[mc.gA(x, y, z + 1)];

    int num = 0;
    if ((d0 < iso && dx >= iso) || (dx < iso && d0 >= iso))
      num++;
    if ((d0 < iso && dy >= iso) || (dy < iso && d0 >= iso))
      num++;
    if ((d0 < iso && dz >= iso) || (dz < iso && d0 >= iso))
      num++;
    mc.used_to_first_mc_vert[used_index] = num;
  }

  template <typename Scalar, typename IndexType>
  inline __device__ Vertex<Scalar> computeMcVert(CuMC<Scalar, IndexType> &mc,
                                                 Scalar const *__restrict__ data, Vertex<Scalar> const *__restrict__ deform, IndexType x,
                                                 IndexType y, IndexType z, int dim, Scalar iso)
  {
    IndexType offset[3] = {0, 0, 0};
    offset[dim] += 1;
    IndexType v0 = mc.gA(x, y, z);
    IndexType v1 = mc.gA(x + offset[0], y + offset[1], z + offset[2]);

    Scalar d0 = data[v0];
    Scalar d1 = data[v1];

    Scalar t = (d1 != d0) ? clamp((iso - d0) / (d1 - d0), Scalar(0.0), Scalar(1.0)) : Scalar(0.5);

    Vertex<Scalar> p0 = {Scalar(x), Scalar(y), Scalar(z)};
    Vertex<Scalar> p1 = {Scalar(x + offset[0]), Scalar(y + offset[1]), Scalar(z + offset[2])};

    if (deform != NULL)
    {
      p0 += deform[v0];
      p1 += deform[v1];
    }

    return p0 + (p1 - p0) * t;
  }

  template <typename Scalar, typename IndexType>
  __global__ void create_cell_mc_verts_kernel(CuMC<Scalar, IndexType> mc,
                                              Scalar const *__restrict__ data,
                                              Vertex<Scalar> const *__restrict__ deform, Scalar iso)
  {

    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= mc.n_used_cells)
      return;
    int cell_index = mc.used_cell_index[used_index];

    IndexType x = mc.gX(cell_index);
    IndexType y = mc.gY(cell_index);
    IndexType z = mc.gZ(cell_index);
    // no need to check range of xyz, already checked when creating used cells

    Scalar d0 = data[cell_index];
    Scalar dx = data[mc.gA(x + 1, y, z)];
    Scalar dy = data[mc.gA(x, y + 1, z)];
    Scalar dz = data[mc.gA(x, y, z + 1)];

    IndexType first = mc.used_to_first_mc_vert[used_index];
    if ((d0 < iso && dx >= iso) || (dx < iso && d0 >= iso))
    {
      auto p = computeMcVert(mc, data, deform, x, y, z, 0, iso);
      mc.verts_type[first] = 0;
      mc.verts[first++] = p;
    }
    if ((d0 < iso && dy >= iso) || (dy < iso && d0 >= iso))
    {
      auto p = computeMcVert(mc, data, deform, x, y, z, 1, iso);
      mc.verts_type[first] = 1;
      mc.verts[first++] = p;
    }
    if ((d0 < iso && dz >= iso) || (dz < iso && d0 >= iso))
    {
      auto p = computeMcVert(mc, data, deform, x, y, z, 2, iso);
      mc.verts_type[first] = 2;
      mc.verts[first++] = p;
    }
  }

  template <typename Scalar, typename IndexType>
  inline __device__ void adjComputeMcVert(CuMC<Scalar, IndexType> &mc,
                                          Scalar const *__restrict__ data, Vertex<Scalar> const *__restrict__ deform, IndexType x, IndexType y,
                                          IndexType z, int dim, Scalar iso,
                                          Scalar *__restrict__ adj_data, Vertex<Scalar> *__restrict__ adj_deform, Vertex<Scalar> adj_ret)
  {
    IndexType offset[3] = {0, 0, 0};
    offset[dim] += 1;
    IndexType v0 = mc.gA(x, y, z);
    IndexType v1 = mc.gA(x + offset[0], y + offset[1], z + offset[2]);
    Scalar d0 = data[v0];
    Scalar d1 = data[v1];
    Scalar t = (d1 != d0) ? clamp((iso - d0) / (d1 - d0), Scalar(0.0), Scalar(1.0)) : Scalar(0.5);

    Vertex<Scalar> p0 = {Scalar(x), Scalar(y), Scalar(z)};
    Vertex<Scalar> p1 = {Scalar(x + offset[0]), Scalar(y + offset[1]), Scalar(z + offset[2])};
    if (deform != NULL)
    {
      p0 += deform[v0];
      p1 += deform[v1];
    }

    Vertex<Scalar> adj_p0 = Vertex<Scalar>{1 - t, 1 - t, 1 - t} * adj_ret;
    Vertex<Scalar> adj_p1 = Vertex<Scalar>{t, t, t} * adj_ret;
    Scalar adj_t = (p1 - p0).dot(adj_ret);

    if (deform != NULL)
    {
      atomicAdd(&adj_deform[v0].x, adj_p0.x);
      atomicAdd(&adj_deform[v0].y, adj_p0.y);
      atomicAdd(&adj_deform[v0].z, adj_p0.z);
      atomicAdd(&adj_deform[v1].x, adj_p1.x);
      atomicAdd(&adj_deform[v1].y, adj_p1.y);
      atomicAdd(&adj_deform[v1].z, adj_p1.z);
    }

    Scalar adj_d0 = (iso - d1) / ((d1 - d0) * (d1 - d0)) * adj_t;
    Scalar adj_d1 = (d0 - iso) / ((d1 - d0) * (d1 - d0)) * adj_t;

    atomicAdd(&adj_data[v0], adj_d0);
    atomicAdd(&adj_data[v1], adj_d1);
  }

  template <typename Scalar, typename IndexType>
  inline __device__ __host__ uint8_t getCellCode(CuMC<Scalar, IndexType> &mc,
                                                 Scalar const *__restrict__ data, IndexType cx,
                                                 IndexType cy, IndexType cz, Scalar iso)
  {
    int code = 0;
    for (int i = 0; i < 8; ++i)
    {
      IndexType cxn = cx + mcCorners[i][0];
      IndexType cyn = cy + mcCorners[i][1];
      IndexType czn = cz + mcCorners[i][2];
      if (data[mc.gA(cxn, cyn, czn)] >= iso)
      {
        code |= (1 << i);
      }
    }
    return code;
  }

  template <typename Scalar, typename IndexType>
  __global__ void adj_create_cell_mc_verts_kernel(CuMC<Scalar, IndexType> mc,
                                              Scalar const *__restrict__ data, Vertex<Scalar> const *__restrict__ deform, Scalar iso,
                                              Scalar *__restrict__ adj_data, Vertex<Scalar> *__restrict__ adj_deform,
                                              Vertex<Scalar> const *__restrict__ adj_verts)
  {

    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= mc.n_used_cells)
      return;
    int cell_index = mc.used_cell_index[used_index];

    IndexType x = mc.gX(cell_index);
    IndexType y = mc.gY(cell_index);
    IndexType z = mc.gZ(cell_index);
    // no need to check range of xyz, already checked when creating used cells

    Scalar d0 = data[cell_index];
    Scalar dx = data[mc.gA(x + 1, y, z)];
    Scalar dy = data[mc.gA(x, y + 1, z)];
    Scalar dz = data[mc.gA(x, y, z + 1)];

    IndexType first = mc.used_to_first_mc_vert[used_index];
    if ((d0 < iso && dx >= iso) || (dx < iso && d0 >= iso))
    {
      Vertex<Scalar> adj_p = adj_verts[first++];
      adjComputeMcVert(mc, data, deform, x, y, z, 0, iso, adj_data, adj_deform, adj_p);
    }
    if ((d0 < iso && dy >= iso) || (dy < iso && d0 >= iso))
    {
      Vertex<Scalar> adj_p = adj_verts[first++];
      adjComputeMcVert(mc, data, deform, x, y, z, 1, iso, adj_data, adj_deform, adj_p);
    }
    if ((d0 < iso && dz >= iso) || (dz < iso && d0 >= iso))
    {
      Vertex<Scalar> adj_p = adj_verts[first++];
      adjComputeMcVert(mc, data, deform, x, y, z, 2, iso, adj_data, adj_deform, adj_p);
    }
  }

  template <typename Scalar, typename IndexType>
  inline __device__ uint8_t getGoodCellCode(CuMC<Scalar, IndexType> &mc,
                                            Scalar const *__restrict__ data, IndexType cx,
                                            IndexType cy, IndexType cz, Scalar iso)
  {
    uint8_t cellCode = getCellCode(mc, data, cx, cy, cz, iso);
    // uint8_t direction = problematicConfigs[cellCode];
    // if (direction != 255)
    // {
    //   IndexType neighborCoords[] = {cx, cy, cz};
    //   uint8_t component = direction >> 1;
    //   int delta = (direction & 1) == 1 ? 1 : -1;
    //   neighborCoords[component] += delta;
    //   if (neighborCoords[component] >= 0 && neighborCoords[component] < (mc.dims[component] - 1))
    //   {
    //     uint8_t neighborCellCode =
    //         getCellCode(mc, data, neighborCoords[0], neighborCoords[1], neighborCoords[2], iso);
    //     if (problematicConfigs[uint8_t(neighborCellCode)] != 255)
    //     {
    //       cellCode ^= 0xff;
    //     }
    //   }
    // }
    return cellCode;
  }

  template <typename Scalar, typename IndexType>
  __global__ void count_cell_mc_tris_kernel(CuMC<Scalar, IndexType> mc,
                                         Scalar const *__restrict__ data, Scalar iso)
  {
    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= mc.n_used_cells)
      return;
    int cell_index = mc.used_cell_index[used_index];

    IndexType x = mc.gX(cell_index);
    IndexType y = mc.gY(cell_index);
    IndexType z = mc.gZ(cell_index);

    mc.used_to_first_mc_tri[used_index] = 0;
    if (x >= mc.dims[0] - 1 || y >= mc.dims[1] - 1 || z >= mc.dims[2] - 1)
      return;

    uint8_t cubeCode = getGoodCellCode(mc, data, x, y, z, iso);
    mc.used_cell_code[used_index] = cubeCode;

    int num = firstMarchingCubesId[static_cast<int>(cubeCode) + 1] - firstMarchingCubesId[cubeCode];
    mc.used_to_first_mc_tri[used_index] = num;
  }

  template <typename Scalar, typename IndexType>
  __global__ void create_cell_mc_tris_kernel(CuMC<Scalar, IndexType> mc,
                                          Scalar const *__restrict__ data, Scalar iso)
  {
    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= mc.n_used_cells)
      return;
    int cell_index = mc.used_cell_index[used_index];

    IndexType x = mc.gX(cell_index);
    IndexType y = mc.gY(cell_index);
    IndexType z = mc.gZ(cell_index);

    if (x >= mc.dims[0] - 1 || y >= mc.dims[1] - 1 || z >= mc.dims[2] - 1)
      return;

    uint8_t cubeCode = mc.used_cell_code[used_index];

    int firstIn = firstMarchingCubesId[cubeCode]; // the starting point of this case
		int num = firstMarchingCubesId[static_cast<int>(cubeCode) + 1] - firstIn; // how many triangles are contained in this cell
		int firstOut = mc.used_to_first_mc_tri[used_index];

    for (int i = 0; i < num; i++)
    {
      int eid = marchingCubesIds[firstIn + i];

      int exi = x + mcEdgeLocations[eid][0];
      int eyi = y + mcEdgeLocations[eid][1];
      int ezi = z + mcEdgeLocations[eid][2];
      int edgeNr = mcEdgeLocations[eid][3];

      int eused_index = mc.first_cell_used[mc.gA(exi, eyi, ezi)];
      int v_num = mc.used_to_first_mc_vert[eused_index+1] - mc.used_to_first_mc_vert[eused_index];
      // printf("v_num %d\n", v_num);
      int id = mc.used_to_first_mc_vert[eused_index];
      for (int k = 0; k < v_num; k++)
      {
        if (mc.verts_type[id+k] == edgeNr)
        {
          // printf("id %d, k %d, mc.verts_type[id+k] %d, edgeNr %d\n", id, k, mc.verts_type[id+k], edgeNr);
          id += k;
          break;
        }
      }
      mc.tris[firstOut + i] = id;
      // printf("i %d, used_index %d, num %d, exi %d, eyi %d, ezi %d, eused_index %d, vert_id %d, edgeNr %d, id %d\n", i, used_index, num, exi, eyi, ezi, eused_index, mc.used_to_first_mc_vert[eused_index], edgeNr, id);
    }
    // int eid, exi, eyi, ezi, edgeNr, eused_index;
    // int i=0;
    // int j=0;
    // while (i<num)
    // {
    //   // t_1
    //   eid = marchingCubesIds[firstIn + i];
    //   exi = x + mcEdgeLocations[eid][0];
    //   eyi = y + mcEdgeLocations[eid][1];
    //   ezi = z + mcEdgeLocations[eid][2];
    //   edgeNr = mcEdgeLocations[eid][3];
    //   eused_index = mc.first_cell_used[mc.gA(exi, eyi, ezi)];
    //   int id_i = mc.used_to_first_mc_vert[used_index] + edgeNr;
    //   i++;
      
    //   // t_2
    //   eid = marchingCubesIds[firstIn + i];
    //   exi = x + mcEdgeLocations[eid][0];
    //   eyi = y + mcEdgeLocations[eid][1];
    //   ezi = z + mcEdgeLocations[eid][2];
    //   edgeNr = mcEdgeLocations[eid][3];
    //   eused_index = mc.first_cell_used[mc.gA(exi, eyi, ezi)];
    //   int id_j = mc.used_to_first_mc_vert[used_index] + edgeNr;
    //   i++;

    //   // t_3
    //   eid = marchingCubesIds[firstIn + i];
    //   exi = x + mcEdgeLocations[eid][0];
    //   eyi = y + mcEdgeLocations[eid][1];
    //   ezi = z + mcEdgeLocations[eid][2];
    //   edgeNr = mcEdgeLocations[eid][3];
    //   eused_index = mc.first_cell_used[mc.gA(exi, eyi, ezi)];
    //   int id_k = mc.used_to_first_mc_vert[used_index] + edgeNr;
    //   i++;

    //   mc.tris[firstOut + j] = {id_i, id_j, id_k};
    //   j++;
    // }
  }

  template <typename Scalar, typename IndexType>
  void CuMC<Scalar, IndexType>::forward(Scalar const *d_data, Vertex<Scalar> const *d_deform, IndexType dimX, IndexType dimY,
                                        IndexType dimZ, Scalar iso)
  {

    resize(dimX, dimY, dimZ);

    size_t temp_storage_bytes = 0;

    ensure_cell_storage_size(n_cells);

    // find and index used cells
    CHECK_CUDA(cudaMemset(first_cell_used + n_cells, 0, sizeof(IndexType)));
    count_used_cells_kernel<<<(n_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this, d_data,
                                                                                     iso);

    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_cell_used, first_cell_used,
                                  n_cells + 1);
    ensure_temp_storage_size(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_cell_used,
                                  first_cell_used, n_cells + 1);

    CHECK_CUDA(cudaMemcpy(&n_used_cells, first_cell_used + n_cells, sizeof(IndexType),
                          cudaMemcpyDeviceToHost));
    // fprintf(stderr, "used cells %d\n", n_used_cells);

    ensure_used_cell_storage_size(n_used_cells);

    CHECK_CUDA(cudaMemset(used_to_first_mc_vert + n_used_cells, 0, sizeof(IndexType)));
    CHECK_CUDA(cudaMemset(used_to_first_mc_tri + n_used_cells, 0, sizeof(IndexType)));

    index_used_cells_kernel<<<(n_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

    // count mc vertices
    count_cell_mc_verts_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, iso);
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, used_to_first_mc_vert,
                                  used_to_first_mc_vert, n_used_cells + 1);
    ensure_temp_storage_size(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, used_to_first_mc_vert,
                                  used_to_first_mc_vert, n_used_cells + 1);
    CHECK_CUDA(cudaMemcpy(&n_verts, used_to_first_mc_vert + n_used_cells, sizeof(IndexType),
                          cudaMemcpyDeviceToHost));

    // // Test used_to_first_mc_vert
    // IndexType *used_to_first_mc_vert_host;
    // cudaMallocHost((void **)&used_to_first_mc_vert_host, n_used_cells * sizeof(IndexType));
    // cudaMemcpy(used_to_first_mc_vert_host, used_to_first_mc_vert, n_used_cells * sizeof(IndexType), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < n_used_cells; i++)
    // {
    //   printf("used_to_first_mc_vert[%d] = %d\n", i, used_to_first_mc_vert_host[i]);
    // }
    // cudaDeviceSynchronize();

    ensure_vert_storage_size(n_verts);
    create_cell_mc_verts_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, d_deform, iso);
      
    // printf("create cell mc verts kernel done %d\n", n_verts);

    // count mc triangles
    count_cell_mc_tris_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, iso);
    // printf("count cell mc tris kernel done\n");
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, used_to_first_mc_tri,
                                  used_to_first_mc_tri, n_used_cells + 1);
    ensure_temp_storage_size(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, used_to_first_mc_tri,
                                  used_to_first_mc_tri, n_used_cells + 1);
    CHECK_CUDA(cudaMemcpy(&n_tris, used_to_first_mc_tri + n_used_cells, sizeof(IndexType),
                          cudaMemcpyDeviceToHost));

    // create mc triangles
    ensure_tri_storage_size(n_tris);
    create_cell_mc_tris_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, iso);
    // printf("create cell mc tris kernel done\n");
    n_tris/=3;

  }

  template <typename Scalar, typename IndexType>
  void CuMC<Scalar, IndexType>::backward(Scalar const *d_data, Vertex<Scalar> const *d_deform, IndexType dimX, IndexType dimY,
                                         IndexType dimZ, Scalar iso, Scalar *adj_d_data, Vertex<Scalar> *adj_d_deform,
                                         Vertex<Scalar> const *adj_verts)
  {
    adj_create_cell_mc_verts_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, d_deform, iso, adj_d_data, adj_d_deform, adj_verts);
  }

  template struct Vertex<float>;
  template struct Vertex<double>;
  template struct Triangle<int>;

  template struct CuMC<double, int>;
  template struct CuMC<float, int>;

} // namespace cumc
