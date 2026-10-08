#include <cuda_runtime.h>
#include <c10/macros/Macros.h>
#include <initializer_list>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "iou_box3d/iou_utils.cuh"
#include "fixture1805.h"

constexpr int kCapacity = 12 * (1 << 6);
struct RawBox {
  const float* data;
  __device__ const float* operator[](int vertex) const {
    return data + 3 * vertex;
  }
  __device__ int size(int) const { return 8; }
};
struct FaceView {
  const FaceVerts* first;
  int first_count;
  const FaceVerts* second;
  __device__ const FaceVerts& operator[](int index) const {
    return index < first_count ? first[index] : second[index - first_count];
  }
};
struct Report {
  int trace[2][7];
  int overflow[2][5]; // plane, source triangle, input count, offset, append count
  int counts[2];
  int retained_second;
  int merged;
  int merge_limit;
  int merge_overflow_index;
  int merge_overflow_second;
  float volume;
  float iou;
};

// The original native ClipTriByPlane is used unchanged. Only the outer
// collection loop has a checked capacity instead of its write-after-clamp.
__device__ int trace_side(
    const FaceVerts* planes,
    const float3& center,
    FaceVerts* faces,
    FaceVerts* scratch,
    int capacity,
    int* trace,
    int* overflow,
    bool assert_on_overflow) {
  int num_tris = NUM_TRIS;
  trace[0] = num_tris;
  for (int plane = 0; plane < NUM_PLANES; ++plane) {
    const float3 normal = PlaneNormalDirection(planes[plane], center);
    int offset = 0;
    for (int triangle = 0; triangle < num_tris; ++triangle) {
      FaceVerts clipped[2];
      int count = ClipTriByPlane(planes[plane], faces[triangle], normal, clipped);
      if (offset + count > capacity) {
        overflow[0] = plane;
        overflow[1] = triangle;
        overflow[2] = num_tris;
        overflow[3] = offset;
        overflow[4] = count;
        if (assert_on_overflow) {
          CUDA_KERNEL_ASSERT(offset + count <= capacity);
        }
        return -1;
      }
      for (int child = 0; child < count; ++child) {
        scratch[offset++] = clipped[child];
      }
    }
    num_tris = offset;
    trace[plane + 1] = num_tris;
    for (int triangle = 0; triangle < num_tris; ++triangle) {
      faces[triangle] = scratch[triangle];
    }
  }
  return num_tris;
}

__global__ void native_counts(
    const float* inputs,
    FaceVerts* storage,
    bool* keep,
    Report* report,
    int capacity,
    bool assert_on_overflow) {
  RawBox box1{inputs};
  RawBox box2{inputs + 24};
  FaceVerts planes1[6], planes2[6];
  FaceVerts original1[12], original2[12];
  FaceVerts* first = storage;
  FaceVerts* second = storage + kCapacity;
  FaceVerts* scratch = storage + 2 * kCapacity;
  GetBoxTris(box1, original1);
  GetBoxTris(box2, original2);
  GetBoxPlanes(box1, planes1);
  GetBoxPlanes(box2, planes2);
  const float3 center1 = BoxCenter(box1);
  const float3 center2 = BoxCenter(box2);
  for (int triangle = 0; triangle < NUM_TRIS; ++triangle) {
    first[triangle] = original1[triangle];
    second[triangle] = original2[triangle];
  }
  report->counts[0] = trace_side(
      planes2, center2, first, scratch, capacity, report->trace[0],
      report->overflow[0], assert_on_overflow);
  report->counts[1] = trace_side(
      planes1, center1, second, scratch, capacity, report->trace[1],
      report->overflow[1], assert_on_overflow);
  report->retained_second = -1;
  report->merged = -1;
  if (report->counts[0] < 0 || report->counts[1] < 0) {
    return;
  }
  const int first_count = report->counts[0];
  const int second_count = report->counts[1];
  for (int triangle = 0; triangle < second_count; ++triangle) {
    keep[triangle] = true;
  }
  for (int first_index = 0; first_index < first_count; ++first_index) {
    for (int second_index = 0; second_index < second_count; ++second_index) {
      if (IsCoplanarTriTri(first[first_index], second[second_index]) &&
          FaceArea(first[first_index]) > aEpsilon) {
        keep[second_index] = false;
      }
    }
  }
  int retained = 0;
  for (int triangle = 0; triangle < second_count; ++triangle) {
    if (keep[triangle]) {
      second[retained++] = second[triangle];
    }
  }
  report->retained_second = retained;
  report->merged = first_count + retained;
  FaceView view{first, first_count, second};
  report->merge_limit = 100;
  for (int triangle = 0; triangle < report->merged; ++triangle) {
    if (triangle >= report->merge_limit) {
      report->merge_overflow_index = triangle;
      report->merge_overflow_second = triangle - first_count;
      break;
    }
    scratch[triangle] = view[triangle];
  }
  report->volume = 0.0f;
  report->iou = 0.0f;
  if (report->merged > 0) {
    const float3 center = PolyhedronCenter(view, report->merged);
    const float volume = BoxVolume(view, center, report->merged);
    const float original_volume1 = BoxVolume(original1, center1, NUM_TRIS);
    const float original_volume2 = BoxVolume(original2, center2, NUM_TRIS);
    report->volume = volume;
    report->iou = volume / (original_volume1 + original_volume2 - volume);
  }
}

static void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    exit(2);
  }
}

int main(int argc, char** argv) {
  int capacity = argc > 1 ? atoi(argv[1]) : kCapacity;
  bool assert_on_overflow = argc > 2 && strcmp(argv[2], "assert") == 0;
  float inputs[48];
  memcpy(inputs, kIssueBoxes, sizeof(inputs));
  if (argc > 3) {
    FILE* input = fopen(argv[3], "rb");
    if (!input || fread(inputs, sizeof(float), 48, input) != 48) {
      fprintf(stderr, "Cannot read 48 float32 coordinates from %s\n", argv[3]);
      return 3;
    }
    fclose(input);
  }
  if (capacity < NUM_TRIS || capacity > kCapacity) {
    fprintf(stderr, "capacity must be in [12,768]\n");
    return 3;
  }
  float* device_inputs;
  FaceVerts* storage;
  bool* keep;
  Report* device_report;
  Report report;
  memset(&report, 0xff, sizeof(report));
  check(cudaMalloc(&device_inputs, sizeof(inputs)), "cudaMalloc inputs");
  check(cudaMalloc(&storage, 3 * kCapacity * sizeof(FaceVerts)), "cudaMalloc faces");
  check(cudaMalloc(&keep, kCapacity * sizeof(bool)), "cudaMalloc keep");
  check(cudaMalloc(&device_report, sizeof(report)), "cudaMalloc report");
  check(cudaMemcpy(device_inputs, inputs, sizeof(inputs), cudaMemcpyHostToDevice),
        "copy inputs");
  check(cudaMemcpy(device_report, &report, sizeof(report), cudaMemcpyHostToDevice),
        "copy report");
  native_counts<<<1, 1>>>(
      device_inputs, storage, keep, device_report, capacity, assert_on_overflow);
  check(cudaGetLastError(), "kernel launch");
  check(cudaDeviceSynchronize(), "kernel synchronization");
  check(cudaMemcpy(&report, device_report, sizeof(report), cudaMemcpyDeviceToHost),
        "copy report result");
  printf("{\"backend\":\"native CUDA helper trace\",\"capacity\":%d,\"trace\":[", capacity);
  for (int side = 0; side < 2; ++side) {
    if (side) printf(",");
    printf("[");
    for (int plane = 0; plane < 7; ++plane) {
      printf("%s%d", plane ? "," : "", report.trace[side][plane]);
    }
    printf("]");
  }
  printf("],\"counts\":[%d,%d],\"retained_second\":%d,\"merged\":%d,\"volume\":%.9g,\"iou\":%.9g,\"overflow\":[",
         report.counts[0], report.counts[1], report.retained_second,
         report.merged, report.volume, report.iou);
  for (int side = 0; side < 2; ++side) {
    if (side) printf(",");
    printf("[");
    for (int field = 0; field < 5; ++field) {
      printf("%s%d", field ? "," : "", report.overflow[side][field]);
    }
    printf("]");
  }
  printf("],\"merge_limit\":%d,\"merge_first_invalid_index\":%d,\"merge_second_index\":%d}\n", report.merge_limit, report.merge_overflow_index, report.merge_overflow_second);
  cudaFree(device_inputs);
  cudaFree(storage);
  cudaFree(keep);
  cudaFree(device_report);
  return 0;
}
