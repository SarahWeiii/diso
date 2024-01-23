#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cub/device/device_scan.cuh>

#include "cudualmc.h"

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

namespace cudualmc
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

  __constant__ int mcCorners[8][3] = {
      {0, 0, 0},
      {1, 0, 0},
      {0, 1, 0},
      {1, 1, 0},
      {0, 0, 1},
      {1, 0, 1},
      {0, 1, 1},
      {1, 1, 1},
  };

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

  __constant__ int mcFirstPatchIndex[257] = {
      0, 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      21, 22, 24, 25, 28, 29, 31, 33, 35, 36, 38, 39, 41, 42, 43, 45, 46, 47, 49,
      51, 53, 54, 56, 59, 60, 61, 63, 65, 66, 67, 68, 69, 70, 71, 73, 74, 76, 77,
      79, 81, 82, 83, 85, 86, 87, 88, 89, 91, 93, 95, 96, 97, 99, 100, 102, 105, 107,
      109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 122, 123, 125, 127, 129, 130, 131, 132,
      133, 134, 136, 139, 141, 143, 145, 147, 149, 150, 153, 157, 159, 161, 163, 165, 166, 167, 168,
      169, 170, 171, 172, 173, 174, 175, 177, 179, 180, 181, 182, 183, 185, 186, 187, 189, 191, 193,
      195, 197, 200, 202, 203, 205, 206, 207, 208, 209, 210, 211, 213, 215, 218, 220, 223, 225, 229,
      231, 233, 235, 237, 238, 240, 241, 243, 244, 245, 247, 248, 249, 251, 253, 255, 256, 257, 259,
      260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 271, 272, 274, 275, 276, 277, 278, 279, 280,
      282, 283, 284, 285, 287, 289, 291, 292, 293, 295, 296, 297, 299, 300, 301, 302, 303, 304, 305,
      306, 307, 309, 310, 311, 312, 314, 315, 316, 317, 318, 320, 321, 322, 323, 324, 325, 327, 328,
      329, 330, 331, 332, 334, 335, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
      350, 351, 352, 353, 354, 355, 356, 357, 358, 358};

  __constant__ int mcFirstEdgeIndex[359] = {
      0, 3, 6, 10, 13, 17, 20, 23, 28, 31, 34, 37, 41, 46, 50, 55,
      60, 64, 67, 71, 74, 77, 82, 85, 88, 93, 96, 99, 102, 108, 111, 114,
      118, 121, 125, 128, 134, 138, 141, 147, 152, 155, 160, 163, 166, 169, 173, 178,
      181, 184, 188, 191, 195, 198, 204, 207, 210, 213, 216, 219, 224, 230, 234, 237,
      242, 245, 251, 256, 260, 265, 270, 274, 277, 281, 287, 292, 295, 300, 303, 307,
      312, 315, 321, 326, 330, 334, 341, 348, 352, 355, 358, 361, 364, 367, 371, 374,
      378, 383, 386, 390, 396, 399, 402, 405, 408, 411, 415, 418, 423, 426, 431, 437,
      443, 448, 452, 457, 460, 464, 470, 475, 479, 482, 487, 492, 495, 499, 504, 507,
      511, 515, 522, 528, 533, 540, 544, 547, 550, 553, 556, 559, 563, 566, 571, 574,
      578, 581, 586, 589, 593, 597, 604, 607, 610, 613, 616, 619, 622, 625, 630, 633,
      639, 642, 647, 650, 656, 659, 666, 672, 677, 683, 689, 694, 700, 705, 712, 716,
      719, 724, 730, 733, 740, 746, 753, 759, 762, 765, 768, 771, 774, 777, 780, 783,
      787, 790, 793, 796, 800, 803, 806, 809, 812, 817, 820, 824, 827, 831, 836, 842,
      847, 853, 859, 864, 867, 870, 874, 877, 880, 883, 886, 891, 894, 897, 900, 903,
      908, 911, 914, 917, 920, 923, 929, 932, 936, 939, 943, 947, 952, 955, 962, 967,
      970, 977, 983, 986, 992, 996, 999, 1003, 1008, 1014, 1017, 1021, 1025, 1029, 1034, 1037,
      1044, 1049, 1052, 1057, 1061, 1066, 1072, 1079, 1084, 1088, 1093, 1099, 1105, 1110, 1113, 1118,
      1125, 1131, 1134, 1140, 1146, 1153, 1158, 1162, 1169, 1172, 1175, 1181, 1184, 1188, 1191, 1195,
      1198, 1202, 1206, 1210, 1215, 1221, 1224, 1229, 1236, 1241, 1244, 1249, 1255, 1262, 1266, 1271,
      1276, 1280, 1285, 1291, 1294, 1299, 1306, 1312, 1317, 1320, 1326, 1332, 1338, 1345, 1352, 1355,
      1358, 1363, 1367, 1373, 1376, 1381, 1384, 1389, 1395, 1402, 1408, 1415, 1422, 1425, 1428, 1434,
      1437, 1443, 1448, 1454, 1459, 1465, 1469, 1472, 1476, 1481, 1486, 1490, 1495, 1499, 1505, 1508,
      1513, 1519, 1523, 1526, 1530, 1533, 1536};

  __constant__ int mcEdgeIndex[1536] = {
      0, 3, 8, 0, 1, 9, 1, 3, 8, 9, 4, 7, 8, 0, 3, 4, 7, 0, 1, 9, 4, 7, 8, 1,
      3, 4, 7, 9, 4, 5, 9, 0, 3, 8, 4, 5, 9, 0, 1, 4, 5, 1, 3, 4, 5, 8, 5, 7,
      8, 9, 0, 3, 5, 7, 9, 0, 1, 5, 7, 8, 1, 3, 5, 7, 2, 3, 11, 0, 2, 8, 11, 0,
      1, 9, 2, 3, 11, 1, 2, 8, 9, 11, 4, 7, 8, 2, 3, 11, 0, 2, 4, 7, 11, 0, 1, 9,
      4, 7, 8, 2, 3, 11, 1, 2, 4, 7, 9, 11, 4, 5, 9, 2, 3, 11, 0, 2, 8, 11, 4, 5,
      9, 0, 1, 4, 5, 2, 3, 11, 1, 2, 4, 5, 8, 11, 5, 7, 8, 9, 2, 3, 11, 0, 2, 5,
      7, 9, 11, 0, 1, 5, 7, 8, 2, 3, 11, 1, 2, 5, 7, 11, 1, 2, 10, 0, 3, 8, 1, 2,
      10, 0, 2, 9, 10, 2, 3, 8, 9, 10, 4, 7, 8, 1, 2, 10, 0, 3, 4, 7, 1, 2, 10, 0,
      2, 9, 10, 4, 7, 8, 2, 3, 4, 7, 9, 10, 4, 5, 9, 1, 2, 10, 0, 3, 8, 4, 5, 9,
      1, 2, 10, 0, 2, 4, 5, 10, 2, 3, 4, 5, 8, 10, 5, 7, 8, 9, 1, 2, 10, 0, 3, 5,
      7, 9, 1, 2, 10, 0, 2, 5, 7, 8, 10, 2, 3, 5, 7, 10, 1, 3, 10, 11, 0, 1, 8, 10,
      11, 0, 3, 9, 10, 11, 8, 9, 10, 11, 4, 7, 8, 1, 3, 10, 11, 0, 1, 4, 7, 10, 11, 0,
      3, 9, 10, 11, 4, 7, 8, 4, 7, 9, 10, 11, 4, 5, 9, 1, 3, 10, 11, 0, 1, 8, 10, 11,
      4, 5, 9, 0, 3, 4, 5, 10, 11, 4, 5, 8, 10, 11, 5, 7, 8, 9, 1, 3, 10, 11, 0, 1,
      5, 7, 9, 10, 11, 0, 3, 5, 7, 8, 10, 11, 5, 7, 10, 11, 6, 7, 11, 0, 3, 8, 6, 7,
      11, 0, 1, 9, 6, 7, 11, 1, 3, 8, 9, 6, 7, 11, 4, 6, 8, 11, 0, 3, 4, 6, 11, 0,
      1, 9, 4, 6, 8, 11, 1, 3, 4, 6, 9, 11, 4, 5, 9, 6, 7, 11, 0, 3, 8, 4, 5, 9,
      6, 7, 11, 0, 1, 4, 5, 6, 7, 11, 1, 3, 4, 5, 8, 6, 7, 11, 5, 6, 8, 9, 11, 0,
      3, 5, 6, 9, 11, 0, 1, 5, 6, 8, 11, 1, 3, 5, 6, 11, 2, 3, 6, 7, 0, 2, 6, 7,
      8, 0, 1, 9, 2, 3, 6, 7, 1, 2, 6, 7, 8, 9, 2, 3, 4, 6, 8, 0, 2, 4, 6, 0,
      1, 9, 2, 3, 4, 6, 8, 1, 2, 4, 6, 9, 4, 5, 9, 2, 3, 6, 7, 0, 2, 6, 7, 8,
      4, 5, 9, 0, 1, 4, 5, 2, 3, 6, 7, 1, 2, 4, 5, 6, 7, 8, 2, 3, 5, 6, 8, 9,
      0, 2, 5, 6, 9, 0, 1, 2, 3, 5, 6, 8, 1, 2, 5, 6, 1, 2, 10, 6, 7, 11, 0, 3,
      8, 1, 2, 10, 6, 7, 11, 0, 2, 9, 10, 6, 7, 11, 2, 3, 8, 9, 10, 6, 7, 11, 4, 6,
      8, 11, 1, 2, 10, 0, 3, 4, 6, 11, 1, 2, 10, 0, 2, 9, 10, 4, 6, 8, 11, 2, 3, 4,
      6, 9, 10, 11, 4, 5, 9, 1, 2, 10, 6, 7, 11, 0, 3, 8, 4, 5, 9, 1, 2, 10, 6, 7,
      11, 0, 2, 4, 5, 10, 6, 7, 11, 2, 3, 4, 5, 8, 10, 6, 7, 11, 5, 6, 8, 9, 11, 1,
      2, 10, 0, 3, 5, 6, 9, 11, 1, 2, 10, 0, 2, 5, 6, 8, 10, 11, 2, 3, 5, 6, 10, 11,
      1, 3, 6, 7, 10, 0, 1, 6, 7, 8, 10, 0, 3, 6, 7, 9, 10, 6, 7, 8, 9, 10, 1, 3,
      4, 6, 8, 10, 0, 1, 4, 6, 10, 0, 3, 4, 6, 8, 9, 10, 4, 6, 9, 10, 4, 5, 9, 1,
      3, 6, 7, 10, 0, 1, 6, 7, 8, 10, 4, 5, 9, 0, 3, 4, 5, 6, 7, 10, 4, 5, 6, 7,
      8, 10, 1, 3, 5, 6, 8, 9, 10, 0, 1, 5, 6, 9, 10, 0, 3, 8, 5, 6, 10, 5, 6, 10,
      5, 6, 10, 0, 3, 8, 5, 6, 10, 0, 1, 9, 5, 6, 10, 1, 3, 8, 9, 5, 6, 10, 4, 7,
      8, 5, 6, 10, 0, 3, 4, 7, 5, 6, 10, 0, 1, 9, 4, 7, 8, 5, 6, 10, 1, 3, 4, 7,
      9, 5, 6, 10, 4, 6, 9, 10, 0, 3, 8, 4, 6, 9, 10, 0, 1, 4, 6, 10, 1, 3, 4, 6,
      8, 10, 6, 7, 8, 9, 10, 0, 3, 6, 7, 9, 10, 0, 1, 6, 7, 8, 10, 1, 3, 6, 7, 10,
      2, 3, 11, 5, 6, 10, 0, 2, 8, 11, 5, 6, 10, 0, 1, 9, 2, 3, 11, 5, 6, 10, 1, 2,
      8, 9, 11, 5, 6, 10, 4, 7, 8, 2, 3, 11, 5, 6, 10, 0, 2, 4, 7, 11, 5, 6, 10, 0,
      1, 9, 4, 7, 8, 2, 3, 11, 5, 6, 10, 1, 2, 4, 7, 9, 11, 5, 6, 10, 4, 6, 9, 10,
      2, 3, 11, 0, 2, 8, 11, 4, 6, 9, 10, 0, 1, 4, 6, 10, 2, 3, 11, 1, 2, 4, 6, 8,
      10, 11, 6, 7, 8, 9, 10, 2, 3, 11, 0, 2, 6, 7, 9, 10, 11, 0, 1, 6, 7, 8, 10, 2,
      3, 11, 1, 2, 6, 7, 10, 11, 1, 2, 5, 6, 0, 3, 8, 1, 2, 5, 6, 0, 2, 5, 6, 9,
      2, 3, 5, 6, 8, 9, 4, 7, 8, 1, 2, 5, 6, 0, 3, 4, 7, 1, 2, 5, 6, 0, 2, 5,
      6, 9, 4, 7, 8, 2, 3, 4, 5, 6, 7, 9, 1, 2, 4, 6, 9, 0, 3, 8, 1, 2, 4, 6,
      9, 0, 2, 4, 6, 2, 3, 4, 6, 8, 1, 2, 6, 7, 8, 9, 0, 1, 2, 3, 6, 7, 9, 0,
      2, 6, 7, 8, 2, 3, 6, 7, 1, 3, 5, 6, 11, 0, 1, 5, 6, 8, 11, 0, 3, 5, 6, 9,
      11, 5, 6, 8, 9, 11, 4, 7, 8, 1, 3, 5, 6, 11, 0, 1, 4, 5, 6, 7, 11, 0, 3, 5,
      6, 9, 11, 4, 7, 8, 4, 5, 6, 7, 9, 11, 1, 3, 4, 6, 9, 11, 0, 1, 4, 6, 8, 9,
      11, 0, 3, 4, 6, 11, 4, 6, 8, 11, 1, 3, 6, 7, 8, 9, 11, 0, 1, 9, 6, 7, 11, 0,
      3, 6, 7, 8, 11, 6, 7, 11, 5, 7, 10, 11, 0, 3, 8, 5, 7, 10, 11, 0, 1, 9, 5, 7,
      10, 11, 1, 3, 8, 9, 5, 7, 10, 11, 4, 5, 8, 10, 11, 0, 3, 4, 5, 10, 11, 0, 1, 9,
      4, 5, 8, 10, 11, 1, 3, 4, 5, 9, 10, 11, 4, 7, 9, 10, 11, 0, 3, 8, 4, 7, 9, 10,
      11, 0, 1, 4, 7, 10, 11, 1, 3, 4, 7, 8, 10, 11, 8, 9, 10, 11, 0, 3, 9, 10, 11, 0,
      1, 8, 10, 11, 1, 3, 10, 11, 2, 3, 5, 7, 10, 0, 2, 5, 7, 8, 10, 0, 1, 9, 2, 3,
      5, 7, 10, 1, 2, 5, 7, 8, 9, 10, 2, 3, 4, 5, 8, 10, 0, 2, 4, 5, 10, 0, 1, 9,
      2, 3, 4, 5, 8, 10, 1, 2, 4, 5, 9, 10, 2, 3, 4, 7, 9, 10, 0, 2, 4, 7, 8, 9,
      10, 0, 1, 2, 3, 4, 7, 10, 4, 7, 8, 1, 2, 10, 2, 3, 8, 9, 10, 0, 2, 9, 10, 0,
      1, 2, 3, 8, 10, 1, 2, 10, 1, 2, 5, 7, 11, 0, 3, 8, 1, 2, 5, 7, 11, 0, 2, 5,
      7, 9, 11, 2, 3, 5, 7, 8, 9, 11, 1, 2, 4, 5, 8, 11, 0, 1, 2, 3, 4, 5, 11, 0,
      2, 4, 5, 8, 9, 11, 4, 5, 9, 2, 3, 11, 1, 2, 4, 7, 9, 11, 0, 3, 8, 1, 2, 4,
      7, 9, 11, 0, 2, 4, 7, 11, 2, 3, 4, 7, 8, 11, 1, 2, 8, 9, 11, 0, 1, 2, 3, 9,
      11, 0, 2, 8, 11, 2, 3, 11, 1, 3, 5, 7, 0, 1, 5, 7, 8, 0, 3, 5, 7, 9, 5, 7,
      8, 9, 1, 3, 4, 5, 8, 0, 1, 4, 5, 0, 3, 4, 5, 8, 9, 4, 5, 9, 1, 3, 4, 7,
      9, 0, 1, 4, 7, 8, 9, 0, 3, 4, 7, 4, 7, 8, 1, 3, 8, 9, 0, 1, 9, 0, 3, 8};

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

  __constant__ int8_t const dmcEdgeOffset[256][12] = {
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1},
      {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1},
      {-1, 0, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1},
      {-1, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1},
      {0, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1},
      {0, 0, -1, -1, 1, -1, -1, 1, 1, 0, -1, -1},
      {-1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1, -1},
      {-1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1},
      {0, -1, -1, 0, 1, 1, -1, -1, 0, 1, -1, -1},
      {0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1},
      {-1, 0, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1},
      {-1, -1, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1},
      {0, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1},
      {0, 0, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1},
      {-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1},
      {-1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0},
      {0, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, 0},
      {0, 0, 1, 1, -1, -1, -1, -1, -1, 0, -1, 1},
      {-1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0},
      {-1, -1, 1, 1, 0, -1, -1, 0, 0, -1, -1, 1},
      {0, -1, 0, -1, 0, -1, -1, 0, -1, -1, -1, 0},
      {0, 0, 2, 2, 1, -1, -1, 1, 1, 0, -1, 2},
      {-1, 0, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0},
      {-1, -1, 1, 1, 0, 0, -1, -1, -1, 0, -1, 1},
      {0, -1, 0, -1, 1, 1, -1, -1, 0, 1, -1, 0},
      {0, 0, 1, 1, 0, 0, -1, -1, -1, -1, -1, 1},
      {-1, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0},
      {-1, -1, 1, 1, -1, 0, -1, 0, 0, 0, -1, 1},
      {0, -1, 0, -1, -1, 0, -1, 0, -1, 0, -1, 0},
      {0, 0, 1, 1, -1, 0, -1, 0, 0, -1, -1, 1},
      {-1, 0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0},
      {-1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1},
      {0, 1, 1, 0, -1, -1, -1, -1, 0, -1, 1, -1},
      {0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1},
      {-1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1},
      {-1, 1, 1, -1, 0, -1, -1, 0, 0, -1, 1, -1},
      {0, 1, 1, 0, 0, -1, -1, 0, -1, -1, 1, -1},
      {0, -1, 0, -1, 1, -1, -1, 1, 1, 0, 0, -1},
      {-1, -1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1},
      {-1, 1, 1, -1, 0, 0, -1, -1, -1, 0, 1, -1},
      {0, 2, 2, 0, 1, 1, -1, -1, 0, 1, 2, -1},
      {0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, -1},
      {-1, -1, 0, 0, 0, 0, -1, -1, 0, -1, 0, -1},
      {-1, 1, 1, -1, -1, 0, -1, 0, 0, 0, 1, -1},
      {0, 1, 1, 0, -1, 0, -1, 0, -1, 0, 1, -1},
      {0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1},
      {-1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1},
      {-1, 0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0},
      {0, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0},
      {0, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0},
      {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0},
      {-1, 1, -1, 1, 0, -1, -1, 0, 0, -1, 1, 1},
      {0, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, 0},
      {0, -1, -1, 0, 1, -1, -1, 1, 1, 0, 0, 0},
      {-1, -1, -1, -1, 0, -1, -1, 0, -1, 0, 0, 0},
      {-1, 1, -1, 1, 0, 0, -1, -1, -1, 0, 1, 1},
      {0, 0, -1, -1, 1, 1, -1, -1, 0, 1, 0, 0},
      {0, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0},
      {-1, -1, -1, -1, 0, 0, -1, -1, 0, -1, 0, 0},
      {-1, 1, -1, 1, -1, 0, -1, 0, 0, 0, 1, 1},
      {0, 0, -1, -1, -1, 0, -1, 0, -1, 0, 0, 0},
      {0, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0},
      {-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0},
      {-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0},
      {0, -1, -1, 0, -1, -1, 1, 1, 0, -1, -1, 1},
      {0, 0, -1, -1, -1, -1, 1, 1, -1, 0, -1, 1},
      {-1, 0, -1, 0, -1, -1, 1, 1, 0, 0, -1, 1},
      {-1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, 0},
      {0, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, 0},
      {0, 0, -1, -1, 1, -1, 1, -1, 1, 0, -1, 1},
      {-1, 0, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0},
      {-1, -1, -1, -1, 0, 0, 1, 1, -1, 0, -1, 1},
      {0, -1, -1, 0, 1, 1, 2, 2, 0, 1, -1, 2},
      {0, 0, -1, -1, 0, 0, 1, 1, -1, -1, -1, 1},
      {-1, 0, -1, 0, 0, 0, 1, 1, 0, -1, -1, 1},
      {-1, -1, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0},
      {0, -1, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0},
      {0, 0, -1, -1, -1, 0, 0, -1, 0, -1, -1, 0},
      {-1, 0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0},
      {-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1},
      {0, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1},
      {0, 0, 1, 1, -1, -1, 1, 1, -1, 0, -1, -1},
      {-1, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1},
      {-1, -1, 0, 0, 0, -1, 0, -1, 0, -1, -1, -1},
      {0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1},
      {0, 0, 1, 1, 1, -1, 1, -1, 1, 0, -1, -1},
      {-1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1, -1},
      {-1, -1, 1, 1, 0, 0, 1, 1, -1, 0, -1, -1},
      {0, -1, 0, -1, 1, 1, 0, 0, 0, 1, -1, -1},
      {0, 0, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1},
      {-1, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, -1},
      {-1, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, -1},
      {0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1, -1},
      {0, 0, 0, 0, -1, 0, 0, -1, 0, -1, -1, -1},
      {-1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1},
      {-1, 0, 0, -1, -1, -1, 1, 1, -1, -1, 0, 1},
      {0, 1, 1, 0, -1, -1, 2, 2, 0, -1, 1, 2},
      {0, -1, 0, -1, -1, -1, 1, 1, -1, 0, 0, 1},
      {-1, -1, 0, 0, -1, -1, 1, 1, 0, 0, 0, 1},
      {-1, 1, 1, -1, 0, -1, 0, -1, 0, -1, 1, 0},
      {0, 1, 1, 0, 0, -1, 0, -1, -1, -1, 1, 0},
      {0, -1, 0, -1, 1, -1, 1, -1, 1, 0, 0, 1},
      {-1, -1, 0, 0, 0, -1, 0, -1, -1, 0, 0, 0},
      {-1, 1, 1, -1, 0, 0, 2, 2, -1, 0, 1, 2},
      {0, 2, 2, 0, 1, 1, 3, 3, 0, 1, 2, 3},
      {0, -1, 0, -1, 0, 0, 1, 1, -1, -1, 0, 1},
      {-1, -1, 0, 0, 0, 0, 1, 1, 0, -1, 0, 1},
      {-1, 1, 1, -1, -1, 0, 0, -1, 0, 0, 1, 0},
      {0, 1, 1, 0, -1, 0, 0, -1, -1, 0, 1, 0},
      {0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, 0},
      {-1, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0},
      {-1, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1},
      {0, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, -1},
      {0, -1, -1, 0, -1, -1, 0, 0, -1, 0, 0, -1},
      {-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1},
      {-1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1},
      {0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1},
      {0, -1, -1, 0, 0, -1, 0, -1, 0, 0, 0, -1},
      {-1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0, -1},
      {-1, 1, -1, 1, 0, 0, 1, 1, -1, 0, 1, -1},
      {0, 0, -1, -1, 1, 1, 0, 0, 0, 1, 0, -1},
      {0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0, -1},
      {-1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, -1},
      {-1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, -1},
      {0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1},
      {0, -1, -1, 0, -1, 1, 1, -1, 0, -1, 1, -1},
      {-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1},
      {-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1},
      {0, -1, -1, 0, -1, 1, 1, -1, 0, -1, 1, -1},
      {0, 0, -1, -1, -1, 1, 1, -1, -1, 0, 1, -1},
      {-1, 0, -1, 0, -1, 1, 1, -1, 0, 0, 1, -1},
      {-1, -1, -1, -1, 0, 1, 1, 0, 0, -1, 1, -1},
      {0, -1, -1, 0, 0, 1, 1, 0, -1, -1, 1, -1},
      {0, 0, -1, -1, 1, 2, 2, 1, 1, 0, 2, -1},
      {-1, 0, -1, 0, 0, 1, 1, 0, -1, 0, 1, -1},
      {-1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0, -1},
      {0, -1, -1, 0, 1, -1, 1, -1, 0, 1, 1, -1},
      {0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1},
      {-1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1},
      {-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1},
      {0, -1, -1, 0, -1, -1, 0, 0, -1, 0, 0, -1},
      {0, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, -1},
      {-1, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1},
      {-1, -1, 0, 0, -1, 1, 1, -1, -1, -1, 1, 0},
      {0, -1, 0, -1, -1, 1, 1, -1, 0, -1, 1, 0},
      {0, 0, 1, 1, -1, 2, 2, -1, -1, 0, 2, 1},
      {-1, 0, 0, -1, -1, 1, 1, -1, 0, 0, 1, 0},
      {-1, -1, 1, 1, 0, 2, 2, 0, 0, -1, 2, 1},
      {0, -1, 0, -1, 0, 1, 1, 0, -1, -1, 1, 0},
      {0, 0, 2, 2, 1, 3, 3, 1, 1, 0, 3, 2},
      {-1, 0, 0, -1, 0, 1, 1, 0, -1, 0, 1, 0},
      {-1, -1, 1, 1, 0, -1, 0, -1, -1, 0, 0, 1},
      {0, -1, 0, -1, 1, -1, 1, -1, 0, 1, 1, 0},
      {0, 0, 1, 1, 0, -1, 0, -1, -1, -1, 0, 1},
      {-1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0},
      {-1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 1},
      {0, -1, 0, -1, -1, -1, 0, 0, -1, 0, 0, 0},
      {0, 0, 1, 1, -1, -1, 0, 0, 0, -1, 0, 1},
      {-1, 0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0},
      {-1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1},
      {0, 1, 1, 0, -1, 1, 1, -1, 0, -1, -1, -1},
      {0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1, -1},
      {-1, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, -1},
      {-1, 1, 1, -1, 0, 1, 1, 0, 0, -1, -1, -1},
      {0, 1, 1, 0, 0, 1, 1, 0, -1, -1, -1, -1},
      {0, -1, 0, -1, 1, 0, 0, 1, 1, 0, -1, -1},
      {-1, -1, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1},
      {-1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1, -1},
      {0, 1, 1, 0, 1, -1, 1, -1, 0, 1, -1, -1},
      {0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1},
      {-1, -1, 0, 0, 0, -1, 0, -1, 0, -1, -1, -1},
      {-1, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1},
      {0, 0, 0, 0, -1, -1, 0, 0, -1, 0, -1, -1},
      {0, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1},
      {-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1},
      {-1, 0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0},
      {0, 0, -1, -1, -1, 0, 0, -1, 0, -1, -1, 0},
      {0, -1, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0},
      {-1, -1, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0},
      {-1, 1, -1, 1, 0, 1, 1, 0, 0, -1, -1, 1},
      {0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0},
      {0, -1, -1, 0, 1, 0, 0, 1, 1, 0, -1, 0},
      {-1, -1, -1, -1, 0, 0, 0, 0, -1, 0, -1, 0},
      {-1, 0, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0},
      {0, 0, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0},
      {0, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, 0},
      {-1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, 0},
      {-1, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1, 0},
      {0, 0, -1, -1, -1, -1, 1, 1, -1, 0, -1, 1},
      {0, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0},
      {-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0},
      {-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0},
      {0, -1, -1, 0, -1, 1, -1, 1, 0, -1, 1, 1},
      {0, 0, -1, -1, -1, 1, -1, 1, -1, 0, 1, 1},
      {-1, 0, -1, 0, -1, 1, -1, 1, 0, 0, 1, 1},
      {-1, -1, -1, -1, 0, 0, -1, -1, 0, -1, 0, 0},
      {0, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0},
      {0, 0, -1, -1, 1, 1, -1, -1, 1, 0, 1, 1},
      {-1, 0, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0},
      {-1, -1, -1, -1, 0, -1, -1, 0, -1, 0, 0, 0},
      {0, -1, -1, 0, 1, -1, -1, 1, 0, 1, 1, 1},
      {0, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, 0},
      {-1, 0, -1, 0, 0, -1, -1, 0, 0, -1, 0, 0},
      {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0},
      {0, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0},
      {0, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0},
      {-1, 0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0},
      {-1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1},
      {0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1},
      {0, 0, 1, 1, -1, 1, -1, 1, -1, 0, 1, -1},
      {-1, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, -1},
      {-1, -1, 0, 0, 0, 0, -1, -1, 0, -1, 0, -1},
      {0, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, -1},
      {0, 0, 1, 1, 1, 1, -1, -1, 1, 0, 1, -1},
      {-1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1},
      {-1, -1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1},
      {0, -1, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1},
      {0, 0, 0, 0, 0, -1, -1, 0, -1, -1, 0, -1},
      {-1, 1, 1, -1, 0, -1, -1, 0, 0, -1, 1, -1},
      {-1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1},
      {0, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1},
      {0, 0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1},
      {-1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1},
      {-1, 0, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0},
      {0, 1, 1, 0, -1, 1, -1, 1, 0, -1, -1, 1},
      {0, -1, 0, -1, -1, 0, -1, 0, -1, 0, -1, 0},
      {-1, -1, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0},
      {-1, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0},
      {0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0},
      {0, -1, 0, -1, 0, 0, -1, -1, 0, 0, -1, 0},
      {-1, -1, 1, 1, 0, 0, -1, -1, -1, 0, -1, 1},
      {-1, 0, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0},
      {0, 1, 1, 0, 1, -1, -1, 1, 0, 1, -1, 1},
      {0, -1, 0, -1, 0, -1, -1, 0, -1, -1, -1, 0},
      {-1, -1, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0},
      {-1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0},
      {0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0},
      {0, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, 0},
      {-1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0},
      {-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1},
      {0, 0, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1},
      {0, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1},
      {-1, -1, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1},
      {-1, 0, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1},
      {0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1},
      {0, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1, -1},
      {-1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1},
      {-1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1, -1},
      {0, 0, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1},
      {0, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1},
      {-1, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1},
      {-1, 0, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1},
      {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1},
      {0, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  };

  // mc edge type to index query
  __constant__ int const dmcQuad[6][4][4] = {
      {{0, 0, 0, 0}, {0, -1, 0, 4}, {0, -1, -1, 6}, {0, 0, -1, 2}},
      {{0, 0, 0, 8}, {0, 0, -1, 11}, {-1, 0, -1, 10}, {-1, 0, 0, 9}},
      {{0, 0, 0, 3}, {-1, 0, 0, 1}, {-1, -1, 0, 5}, {0, -1, 0, 7}},

      {{0, 0, 0, 0}, {0, 0, -1, 2}, {0, -1, -1, 6}, {0, -1, 0, 4}},
      {{0, 0, 0, 8}, {-1, 0, 0, 9}, {-1, 0, -1, 10}, {0, 0, -1, 11}},
      {{0, 0, 0, 3}, {0, -1, 0, 7}, {-1, -1, 0, 5}, {-1, 0, 0, 1}},

  };

  template <typename Scalar, typename IndexType>
  void CUDualMC<Scalar, IndexType>::ensure_temp_storage_size(size_t size)
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
  void CUDualMC<Scalar, IndexType>::ensure_cell_storage_size(size_t cell_count)
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
  void CUDualMC<Scalar, IndexType>::ensure_used_cell_storage_size(size_t cell_count)
  {
    if (cell_count > allocated_used_cell_count)
    {
      allocated_used_cell_count = size_t(cell_count + cell_count / 5);
      // fprintf(stderr, "allocated_used_cell_count %ld\n", allocated_used_cell_count);

      CHECK_CUDA(cudaFree(used_to_first_mc_vert));
      CHECK_CUDA(cudaFree(used_to_first_mc_patch));
      CHECK_CUDA(cudaFree(used_cell_code));
      CHECK_CUDA(cudaFree(used_cell_index));

      // TODO
      // CHECK_CUDA(cudaFree(used_cell_mc_vert));
      // CHECK_CUDA(
      //     cudaMalloc((void **)&used_cell_mc_vert, allocated_used_cell_count * 3 *
      //     sizeof(IndexType)));

      CHECK_CUDA(cudaMalloc((void **)&used_to_first_mc_vert,
                            (allocated_used_cell_count + 1) * sizeof(IndexType)));
      CHECK_CUDA(cudaMalloc((void **)&used_to_first_mc_patch,
                            (allocated_used_cell_count + 1) * sizeof(IndexType)));
      CHECK_CUDA(cudaMalloc((void **)&used_cell_code, allocated_used_cell_count * sizeof(uint8_t)));
      CHECK_CUDA(
          cudaMalloc((void **)&used_cell_index, allocated_used_cell_count * sizeof(IndexType)));
    }
  }

  template <typename Scalar, typename IndexType>
  void CUDualMC<Scalar, IndexType>::ensure_quad_storage_size(size_t quad_count)
  {
    if (quad_count > allocated_quad_count)
    {
      allocated_quad_count = size_t(quad_count + quad_count / 5);
      // fprintf(stderr, "allocated_quad_count %ld\n", allocated_quad_count);

      CHECK_CUDA(cudaFree(mc_vert_to_cell));
      CHECK_CUDA(cudaFree(mc_vert_type));
      CHECK_CUDA(cudaFree(quads));

      // TODO
      // CHECK_CUDA(cudaFree(mc_verts));
      // CHECK_CUDA(cudaMalloc((void **)&mc_verts, allocated_quad_count * sizeof(Vertex<Scalar>)));

      CHECK_CUDA(cudaMalloc((void **)&mc_vert_to_cell, allocated_quad_count * sizeof(IndexType)));
      CHECK_CUDA(cudaMalloc((void **)&mc_vert_type, allocated_quad_count * sizeof(uint8_t)));
      CHECK_CUDA(cudaMalloc((void **)&quads, allocated_quad_count * sizeof(Quad<IndexType>)));
    }
  }

  template <typename Scalar, typename IndexType>
  void CUDualMC<Scalar, IndexType>::ensure_vert_storage_size(size_t vert_count)
  {
    if (vert_count > allocated_vert_count)
    {
      allocated_vert_count = size_t(vert_count + vert_count / 5);
      // fprintf(stderr, "allocated_vert_count %ld\n", allocated_vert_count);

      CHECK_CUDA(cudaFree(verts));

      CHECK_CUDA(cudaMalloc((void **)&verts, allocated_vert_count * sizeof(Vertex<Scalar>)));
    }
  }

  template <typename Scalar, typename IndexType>
  __global__ void count_used_cells_kernel(CUDualMC<Scalar, IndexType> dmc,
                                          Scalar const *__restrict__ data, Scalar iso)
  {
    int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= dmc.n_cells)
      return;

    IndexType x = dmc.gX(cell_index);
    IndexType y = dmc.gY(cell_index);
    IndexType z = dmc.gZ(cell_index);

    dmc.first_cell_used[cell_index] = 0;
    if (x >= dmc.dims[0] - 1 || y >= dmc.dims[1] - 1 || z >= dmc.dims[2] - 1)
      return;

    int code = 0;
    for (int i = 0; i < 8; ++i)
    {
      IndexType cxn = x + mcCorners[i][0];
      IndexType cyn = y + mcCorners[i][1];
      IndexType czn = z + mcCorners[i][2];
      if (data[dmc.gA(cxn, cyn, czn)] >= iso)
      {
        code |= (1 << i);
      }
    }

    if (code != 0 && code != 255)
    {
      dmc.first_cell_used[cell_index] = 1;
    }
  }

  template <typename Scalar, typename IndexType>
  __global__ void index_used_cells_kernel(CUDualMC<Scalar, IndexType> dmc)
  {
    int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= dmc.n_cells)
      return;

    int used_index = dmc.first_cell_used[cell_index];
    if (dmc.first_cell_used[cell_index + 1] - used_index)
    {
      dmc.used_cell_index[used_index] = cell_index;
    }
  }

  template <typename Scalar, typename IndexType>
  __global__ void count_cell_mc_verts_kernel(CUDualMC<Scalar, IndexType> dmc,
                                             Scalar const *__restrict__ data, Scalar iso)
  {

    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= dmc.n_used_cells)
      return;
    int cell_index = dmc.used_cell_index[used_index];

    IndexType x = dmc.gX(cell_index);
    IndexType y = dmc.gY(cell_index);
    IndexType z = dmc.gZ(cell_index);
    // no need to check range of xyz, already checked when creating used cells

    Scalar d0 = data[cell_index];
    Scalar dx = data[dmc.gA(x + 1, y, z)];
    Scalar dy = data[dmc.gA(x, y + 1, z)];
    Scalar dz = data[dmc.gA(x, y, z + 1)];

    int num = 0;
    if ((d0 < iso && dx >= iso) || (dx < iso && d0 >= iso))
      num++;
    if ((d0 < iso && dy >= iso) || (dy < iso && d0 >= iso))
      num++;
    if ((d0 < iso && dz >= iso) || (dz < iso && d0 >= iso))
      num++;
    dmc.used_to_first_mc_vert[used_index] = num;
  }

  template <typename Scalar, typename IndexType>
  inline __device__ Vertex<Scalar> computeMcVert(CUDualMC<Scalar, IndexType> &dmc,
                                                 Scalar const *__restrict__ data, Vertex<Scalar> const *__restrict__ deform, IndexType x,
                                                 IndexType y, IndexType z, int dim, Scalar iso)
  {
    IndexType offset[3] = {0, 0, 0};
    offset[dim] += 1;
    IndexType v0 = dmc.gA(x, y, z);
    IndexType v1 = dmc.gA(x + offset[0], y + offset[1], z + offset[2]);

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
  inline __device__ void adjComputeMcVert(CUDualMC<Scalar, IndexType> &dmc,
                                          Scalar const *__restrict__ data, Vertex<Scalar> const *__restrict__ deform, IndexType x, IndexType y,
                                          IndexType z, int dim, Scalar iso,
                                          Scalar *__restrict__ adj_data, Vertex<Scalar> *__restrict__ adj_deform, Vertex<Scalar> adj_ret)
  {
    IndexType offset[3] = {0, 0, 0};
    offset[dim] += 1;
    IndexType v0 = dmc.gA(x, y, z);
    IndexType v1 = dmc.gA(x + offset[0], y + offset[1], z + offset[2]);
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

    Vertex<Scalar> adj_p0 = Vertex<Scalar>{1-t, 1-t, 1-t}*adj_ret;
    Vertex<Scalar> adj_p1 = Vertex<Scalar>{t, t, t}*adj_ret;
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
  __global__ void index_cell_mc_verts_kernel(CUDualMC<Scalar, IndexType> dmc,
                                             Scalar const *__restrict__ data, Scalar iso)
  {
    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= dmc.n_used_cells)
      return;
    int cell_index = dmc.used_cell_index[used_index];

    IndexType x = dmc.gX(cell_index);
    IndexType y = dmc.gY(cell_index);
    IndexType z = dmc.gZ(cell_index);

    if (x >= dmc.dims[0] - 1 || y >= dmc.dims[1] - 1 || z >= dmc.dims[2] - 1)
      return;

    Scalar d0 = data[cell_index];
    Scalar ds[3];
    ds[0] = data[dmc.gA(x + 1, y, z)];
    ds[1] = data[dmc.gA(x, y + 1, z)];
    ds[2] = data[dmc.gA(x, y, z + 1)];

    IndexType first = dmc.used_to_first_mc_vert[used_index];

    for (int dim = 0; dim < 3; dim++)
    {
      Scalar d = ds[dim];
      // dmc.used_cell_mc_vert[3 * used_index + dim] = 0;

      bool entering = d0 < iso && d >= iso;
      bool exiting = d < iso && d0 >= iso;
      if (entering || exiting)
      {
        IndexType id = first++;
        dmc.mc_vert_to_cell[id] = used_index;
        dmc.mc_vert_type[id] = (exiting ? 3 : 0) + dim;

        // dmc.used_cell_mc_vert[3 * used_index + dim] = id;
        // dmc.mc_verts[id] = computeMcVert(dmc, data, x, y, z, dim, iso);
      }
    }
  }

  template <typename Scalar, typename IndexType>
  inline __device__ __host__ uint8_t getCellCode(CUDualMC<Scalar, IndexType> &dmc,
                                                 Scalar const *__restrict__ data, IndexType cx,
                                                 IndexType cy, IndexType cz, Scalar iso)
  {
    int code = 0;
    for (int i = 0; i < 8; ++i)
    {
      IndexType cxn = cx + mcCorners[i][0];
      IndexType cyn = cy + mcCorners[i][1];
      IndexType czn = cz + mcCorners[i][2];
      if (data[dmc.gA(cxn, cyn, czn)] >= iso)
      {
        code |= (1 << i);
      }
    }
    return code;
  }

  template <typename Scalar, typename IndexType>
  inline __device__ uint8_t getGoodCellCode(CUDualMC<Scalar, IndexType> &dmc,
                                            Scalar const *__restrict__ data, IndexType cx,
                                            IndexType cy, IndexType cz, Scalar iso)
  {
    uint8_t cellCode = getCellCode(dmc, data, cx, cy, cz, iso);
    uint8_t direction = problematicConfigs[cellCode];
    if (direction != 255)
    {
      IndexType neighborCoords[] = {cx, cy, cz};
      uint8_t component = direction >> 1;
      int delta = (direction & 1) == 1 ? 1 : -1;
      neighborCoords[component] += delta;
      if (neighborCoords[component] >= 0 && neighborCoords[component] < (dmc.dims[component] - 1))
      {
        uint8_t neighborCellCode =
            getCellCode(dmc, data, neighborCoords[0], neighborCoords[1], neighborCoords[2], iso);
        if (problematicConfigs[uint8_t(neighborCellCode)] != 255)
        {
          cellCode ^= 0xff;
        }
      }
    }
    return cellCode;
  }

  template <typename Scalar, typename IndexType>
  __global__ void count_cell_patches_kernel(CUDualMC<Scalar, IndexType> dmc,
                                            Scalar const *__restrict__ data, Scalar iso)
  {
    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= dmc.n_used_cells)
      return;
    int cell_index = dmc.used_cell_index[used_index];

    IndexType x = dmc.gX(cell_index);
    IndexType y = dmc.gY(cell_index);
    IndexType z = dmc.gZ(cell_index);

    dmc.used_to_first_mc_patch[used_index] = 0;
    if (x >= dmc.dims[0] - 1 || y >= dmc.dims[1] - 1 || z >= dmc.dims[2] - 1)
      return;

    uint8_t cubeCode = getGoodCellCode(dmc, data, x, y, z, iso);
    dmc.used_cell_code[used_index] = cubeCode;

    int num = mcFirstPatchIndex[static_cast<int>(cubeCode) + 1] - mcFirstPatchIndex[cubeCode];
    dmc.used_to_first_mc_patch[used_index] = num;
  }

  // TODO: combine count for speedup
  // template <typename Scalar, typename IndexType>
  // __global__ void
  // count_cell_mc_verts_and_patches_kernel(CUDualMC<Scalar, IndexType> dmc,
  //                                        Scalar const *__restrict__ data,
  //                                        Scalar iso, int n_used) {

  //   int used_index = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (used_index >= n_used)
  //     return;
  //   int cell_index = dmc.used_cell_index[used_index];

  //   IndexType x = dmc.gX(cell_index);
  //   IndexType y = dmc.gY(cell_index);
  //   IndexType z = dmc.gZ(cell_index);

  //   dmc.used_to_first_mc_vert[used_index] = 0;
  //   if (x >= dmc.dims[0] - 1 || y >= dmc.dims[1] - 1 || z >= dmc.dims[2] - 1)
  //     return;

  //   Scalar d0 = data[cell_index];
  //   Scalar dx = data[dmc.gA(x + 1, y, z)];
  //   Scalar dy = data[dmc.gA(x, y + 1, z)];
  //   Scalar dz = data[dmc.gA(x, y, z + 1)];

  //   uint8_t cubeCode = getGoodCellCode(dmc, data, x, y, z, iso);
  //   dmc.used_cell_code[used_index] = cubeCode;

  //   int num_patches = mcFirstPatchIndex[static_cast<int>(cubeCode) + 1] -
  //                     mcFirstPatchIndex[cubeCode];
  //   dmc.used_to_first_mc_patch[used_index] = num_patches;

  //   int num_verts = 0;
  //   if ((d0 <= iso && dx >= iso) || (dx <= iso && d0 >= iso))
  //     num_verts++;
  //   if ((d0 <= iso && dy >= iso) || (dy <= iso && d0 >= iso))
  //     num_verts++;
  //   if ((d0 <= iso && dz >= iso) || (dz <= iso && d0 >= iso))
  //     num_verts++;
  //   dmc.used_to_first_mc_vert[used_index] = num_verts;
  // }

  template <typename Scalar, typename IndexType>
  __global__ void create_dmc_verts_kernel(CUDualMC<Scalar, IndexType> dmc,
                                          Scalar const *__restrict__ data, Vertex<Scalar> const *__restrict__ deform, Scalar iso)
  {
    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= dmc.n_used_cells)
      return;
    int cell_index = dmc.used_cell_index[used_index];

    IndexType x = dmc.gX(cell_index);
    IndexType y = dmc.gY(cell_index);
    IndexType z = dmc.gZ(cell_index);

    if (x >= dmc.dims[0] - 1 || y >= dmc.dims[1] - 1 || z >= dmc.dims[2] - 1)
      return;

    IndexType first = dmc.used_to_first_mc_patch[used_index];
    uint8_t cubeCode = dmc.used_cell_code[used_index];

    for (int patch_index = mcFirstPatchIndex[cubeCode];
         patch_index < mcFirstPatchIndex[static_cast<int>(cubeCode) + 1]; ++patch_index)
    {

      Vertex<Scalar> p{0, 0, 0};
      Scalar num{0};

      for (int edge_index = mcFirstEdgeIndex[patch_index];
           edge_index < mcFirstEdgeIndex[patch_index + 1]; ++edge_index)
      {
        int eid = mcEdgeIndex[edge_index];

        int ex = mcEdgeLocations[eid][0];
        int ey = mcEdgeLocations[eid][1];
        int ez = mcEdgeLocations[eid][2];
        int en = mcEdgeLocations[eid][3];

        // TODO: potential speedup
        // int vid = dmc.used_cell_mc_vert[3 * dmc.first_cell_used[dmc.gA(x + ex, y + ey, z + ez)] +
        // en]; p += dmc.mc_verts[vid];

        // NOTE: comptue mc verts on the fly
        p += computeMcVert(dmc, data, deform, x + ex, y + ey, z + ez, en, iso);
        num += Scalar(1);
      }
      p *= Scalar(1) / num;

      dmc.verts[first++] = p;
    }
  }

  template <typename Scalar, typename IndexType>
  __global__ void adj_create_dmc_verts_kernel(CUDualMC<Scalar, IndexType> dmc,
                                              Scalar const *__restrict__ data, Vertex<Scalar> const *__restrict__ deform, Scalar iso,
                                              Scalar *__restrict__ adj_data, Vertex<Scalar> *__restrict__ adj_deform,
                                              Vertex<Scalar> const *__restrict__ adj_verts)
  {
    int used_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (used_index >= dmc.n_used_cells)
      return;
    int cell_index = dmc.used_cell_index[used_index];

    IndexType x = dmc.gX(cell_index);
    IndexType y = dmc.gY(cell_index);
    IndexType z = dmc.gZ(cell_index);

    if (x >= dmc.dims[0] - 1 || y >= dmc.dims[1] - 1 || z >= dmc.dims[2] - 1)
      return;

    IndexType first = dmc.used_to_first_mc_patch[used_index];
    uint8_t cubeCode = dmc.used_cell_code[used_index];

    for (int patch_index = mcFirstPatchIndex[cubeCode];
         patch_index < mcFirstPatchIndex[static_cast<int>(cubeCode) + 1]; ++patch_index)
    {

      Scalar num{0};
      for (int edge_index = mcFirstEdgeIndex[patch_index];
           edge_index < mcFirstEdgeIndex[patch_index + 1]; ++edge_index)
      {
        num += Scalar(1);
      }

      // backward
      Vertex<Scalar> adj_p = adj_verts[first];
      adj_p *= (Scalar(1) / num);

      for (int edge_index = mcFirstEdgeIndex[patch_index];
           edge_index < mcFirstEdgeIndex[patch_index + 1]; ++edge_index)
      {
        int eid = mcEdgeIndex[edge_index];
        int ex = mcEdgeLocations[eid][0];
        int ey = mcEdgeLocations[eid][1];
        int ez = mcEdgeLocations[eid][2];
        int en = mcEdgeLocations[eid][3];

        adjComputeMcVert(dmc, data, deform, x + ex, y + ey, z + ez, en, iso, adj_data, adj_deform, adj_p);
      }
    }
  }

  template <typename Scalar>
  __device__ Scalar sigmoid(Scalar x)
  {
    return 1.0 / (1 + exp(-x));
  }

  template <typename Scalar>
  __device__ Scalar adj_sigmoid(Scalar x)
  {
    return sigmoid(x) * (1-sigmoid(x));
  }

  template <typename Scalar, typename IndexType>
  __device__ int get_quad_index(CUDualMC<Scalar, IndexType> &dmc, IndexType used_index, int eid)
  {
    uint8_t cellCode = dmc.used_cell_code[used_index];
    int offset = dmcEdgeOffset[cellCode][eid];
    return dmc.used_to_first_mc_patch[used_index] + offset;
  }

  template <typename Scalar, typename IndexType>
  __global__ void create_quads_kernel(CUDualMC<Scalar, IndexType> dmc)
  {
    // launch with n_quads == n_mc_verts
    int quad_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad_index >= dmc.n_quads)
      return;

    IndexType used_index = dmc.mc_vert_to_cell[quad_index];
    IndexType cell_index = dmc.used_cell_index[used_index];

    IndexType x = dmc.gX(cell_index);
    IndexType y = dmc.gY(cell_index);
    IndexType z = dmc.gZ(cell_index);

    uint8_t vert_type = dmc.mc_vert_type[quad_index];

    Quad<IndexType> q;
    for (int i = 0; i < 4; ++i)
    {
      int dx = dmcQuad[vert_type][i][0];
      int dy = dmcQuad[vert_type][i][1];
      int dz = dmcQuad[vert_type][i][2];
      int eid = dmcQuad[vert_type][i][3];

      q.data_ptr()[i] =
          get_quad_index(dmc, dmc.first_cell_used[dmc.gA(x + dx, y + dy, z + dz)], eid);
    }
    dmc.quads[quad_index] = q;
  }

  template <typename Scalar, typename IndexType>
  void CUDualMC<Scalar, IndexType>::forward(Scalar const *d_data, Vertex<Scalar> const *d_deform, IndexType dimX, IndexType dimY,
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
    CHECK_CUDA(cudaMemset(used_to_first_mc_patch + n_used_cells, 0, sizeof(IndexType)));

    index_used_cells_kernel<<<(n_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

    // count mc vertices (dmc patches)
    count_cell_mc_verts_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, iso);
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, used_to_first_mc_vert,
                                  used_to_first_mc_vert, n_used_cells + 1);
    ensure_temp_storage_size(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, used_to_first_mc_vert,
                                  used_to_first_mc_vert, n_used_cells + 1);
    CHECK_CUDA(cudaMemcpy(&n_quads, used_to_first_mc_vert + n_used_cells, sizeof(IndexType),
                          cudaMemcpyDeviceToHost));
    // fprintf(stderr, "mc vert (dmc quad) count: %d\n", n_quads);

    ensure_quad_storage_size(n_quads);

    // index mc vertices (dmc quads)
    index_cell_mc_verts_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, iso);

    // count mc patches (dmc verts)
    count_cell_patches_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, iso);

    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, used_to_first_mc_patch,
                                  used_to_first_mc_patch, n_used_cells + 1);
    ensure_temp_storage_size(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, used_to_first_mc_patch,
                                  used_to_first_mc_patch, n_used_cells + 1);

    CHECK_CUDA(cudaMemcpy(&n_verts, used_to_first_mc_patch + n_used_cells, sizeof(IndexType),
                          cudaMemcpyDeviceToHost));
    ensure_vert_storage_size(n_verts);

    create_dmc_verts_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, d_deform, iso);

    create_quads_kernel<<<(n_quads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
  }

  template <typename Scalar, typename IndexType>
  void CUDualMC<Scalar, IndexType>::backward(Scalar const *d_data, Vertex<Scalar> const *d_deform, IndexType dimX, IndexType dimY,
                                             IndexType dimZ, Scalar iso, Scalar *adj_d_data, Vertex<Scalar> *adj_d_deform,
                                             Vertex<Scalar> const *adj_verts)
  {
    adj_create_dmc_verts_kernel<<<(n_used_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        *this, d_data, d_deform, iso, adj_d_data, adj_d_deform, adj_verts);      
  }

  template struct Vertex<float>;
  template struct Vertex<double>;
  template struct Quad<int>;

  template struct CUDualMC<double, int>;
  template struct CUDualMC<float, int>;

} // namespace cudualmc
