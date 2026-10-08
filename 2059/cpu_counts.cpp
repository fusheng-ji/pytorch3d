#include "iou_box3d/iou_utils.h"
#include "fixture1805.h"
#include <cstdio>
#include <cstring>

struct RawBox {
  const float (*data)[3];
  const float* operator[](int vertex) const { return data[vertex]; }
};

int main(int argc, char** argv) {
  float inputs[2][8][3];
  memcpy(inputs, kIssueBoxes, sizeof(inputs));
  if (argc > 1) {
    FILE* input = fopen(argv[1], "rb");
    if (!input || fread(inputs, sizeof(float), 48, input) != 48) return 3;
    fclose(input);
  }
  face_verts faces[2];
  face_verts planes[2];
  vec3<float> centers[2] = {{0, 0, 0}, {0, 0, 0}};
  int trace[2][7];
  for (int side = 0; side < 2; ++side) {
    RawBox box{inputs[side]};
    faces[side] = GetBoxTris(box);
    planes[side] = GetBoxPlanes(box);
    auto tensor = at::from_blob(inputs[side], {8, 3}, at::kFloat);
    centers[side] = BoxCenter(tensor);
  }
  const float volume1 = BoxVolume(faces[0], centers[0]);
  const float volume2 = BoxVolume(faces[1], centers[1]);
  for (int side = 0; side < 2; ++side) {
    const int other = 1 - side;
    const face_verts direct = BoxIntersections(faces[side], planes[other], centers[other]);
    trace[side][0] = faces[side].size();
    for (int plane = 0; plane < 6; ++plane) {
      const vec3<float> normal = PlaneNormalDirection(planes[other][plane], centers[other]);
      face_verts updated;
      for (const auto& triangle : faces[side]) {
        const auto clipped = ClipTriByPlane(planes[other][plane], triangle, normal);
        updated.insert(updated.end(), clipped.begin(), clipped.end());
      }
      faces[side] = updated;
      trace[side][plane + 1] = updated.size();
    }
    if (direct.size() != faces[side].size()) return 4;
  }
  int counts[2] = {static_cast<int>(faces[0].size()), static_cast<int>(faces[1].size())};
  std::vector<int> keep(faces[1].size(), 1);
  for (const auto& first : faces[0]) {
    for (size_t second = 0; second < faces[1].size(); ++second) {
      if (IsCoplanarTriTri(first, faces[1][second]) && FaceArea(first) > aEpsilon) {
        keep[second] = 0;
      }
    }
  }
  int retained = 0;
  for (size_t second = 0; second < faces[1].size(); ++second) {
    if (keep[second]) {
      faces[0].push_back(faces[1][second]);
      ++retained;
    }
  }
  float volume = 0.0f, iou = 0.0f;
  if (!faces[0].empty()) {
    auto center = PolyhedronCenter(faces[0]);
    volume = BoxVolume(faces[0], center);
    iou = volume / (volume1 + volume2 - volume);
  }
  printf("{\"backend\":\"native CPU helpers\",\"trace\":[");
  for (int side = 0; side < 2; ++side) {
    printf("%s[", side ? "," : "");
    for (int plane = 0; plane < 7; ++plane) {
      printf("%s%d", plane ? "," : "", trace[side][plane]);
    }
    printf("]");
  }
  printf("],\"counts\":[%d,%d],\"retained_second\":%d,\"merged\":%zu,\"volume\":%.9g,\"iou\":%.9g}\n",
      counts[0], counts[1], retained, faces[0].size(), volume, iou);
}
