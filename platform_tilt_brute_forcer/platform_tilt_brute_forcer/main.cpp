#include <fstream>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Path.hpp"
#include "vmath.hpp"

std::ofstream out_stream;

__global__ void cudaFunc(bool* out, const float minX, const float deltaX, const float minZ, const float deltaZ, const int width, const int height, float normalX, float normalY, float normalZ, int frames) {
    const float platformPos[3] = { -1945.0f, -3225.0f, -715.0f };
    const short defaultTriangles[2][3][3] = { {{307, 307, -306}, {-306, 307, -306}, {-306, 307, 307}}, {{307, 307, -306}, {-306, 307, 307}, {307, 307, 307}} };

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > width * height) {
        return;
    }

    float marioPos[3] = { minX + deltaX * (idx%width), -2500.0f, minZ + deltaZ * (idx/width) };
    float normal[3] = { normalX, normalY, normalZ };

    out[idx] = false;

    float mat[4][4];
    mat[1][0] = (normal[0] <= 0.0f) ? ((0.0f - normal[0] < 0.01f) ? 0.0f : (normal[0] + 0.01f)) : ((0.0f - normal[0] > -0.01f) ? 0.0f : (normal[0] - 0.01f));
    mat[1][1] = (normal[1] <= 1.0f) ? ((1.0f - normal[1] < 0.01f) ? 1.0f : (normal[1] + 0.01f)) : ((1.0f - normal[1] > -0.01f) ? 1.0f : (normal[1] - 0.01f));
    mat[1][2] = (normal[2] <= 0.0f) ? ((0.0f - normal[2] < 0.01f) ? 0.0f : (normal[2] + 0.01f)) : ((0.0f - normal[2] > -0.01f) ? 0.0f : (normal[2] - 0.01f));

    float invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

    mat[1][0] *= invsqrt;
    mat[1][1] *= invsqrt;
    mat[1][2] *= invsqrt;

    mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
    mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
    mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

    invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

    mat[0][0] *= invsqrt;
    mat[0][1] *= invsqrt;
    mat[0][2] *= invsqrt;

    mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
    mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
    mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

    invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

    mat[2][0] *= invsqrt;
    mat[2][1] *= invsqrt;
    mat[2][2] *= invsqrt;

    mat[3][0] = platformPos[0];
    mat[3][1] = platformPos[1];
    mat[3][2] = platformPos[2];
    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    short currentTriangles[2][3][3];
    float triangleNormals[2][3];
    float originOffsets[2];

    for (int h = 0; h < 2; h++) {
        for (int i = 0; i < 3; i++) {
            float vx = defaultTriangles[h][i][0];
            float vy = defaultTriangles[h][i][1];
            float vz = defaultTriangles[h][i][2];

            currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
            currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
            currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
        }

        triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
        triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
        triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

        invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

        triangleNormals[h][0] *= invsqrt;
        triangleNormals[h][1] *= invsqrt;
        triangleNormals[h][2] *= invsqrt;

        originOffsets[h] = -(triangleNormals[h][0] * currentTriangles[h][0][0] + triangleNormals[h][1] * currentTriangles[h][0][1] + triangleNormals[h][2] * currentTriangles[h][0][2]);
    }

    float floor_height = 0.0;
    int floor_idx = -1;

    short x = (short)(int)marioPos[0];
    short y = (short)(int)marioPos[1];
    short z = (short)(int)marioPos[2];

    for (int i = 0; i < 2; i++) {
        short x1 = currentTriangles[i][0][0];
        short z1 = currentTriangles[i][0][2];
        short x2 = currentTriangles[i][1][0];
        short z2 = currentTriangles[i][1][2];

        // Check that the point is within the triangle bounds.
        if ((z1 - z) * (x2 - x1) - (x1 - x) * (z2 - z1) < 0) {
            continue;
        }

        // To slightly save on computation time, set this later.
        int16_t x3 = currentTriangles[i][2][0];
        int16_t z3 = currentTriangles[i][2][2];

        if ((z2 - z) * (x3 - x2) - (x2 - x) * (z3 - z2) < 0) {
            continue;
        }
        if ((z3 - z) * (x1 - x3) - (x3 - x) * (z1 - z3) < 0) {
            continue;
        }

        float nx = triangleNormals[i][0];
        float ny = triangleNormals[i][1];
        float nz = triangleNormals[i][2];
        float oo = -(nx * x1 + ny * currentTriangles[i][0][1] + nz * z1);

        // Find the height of the floor at a given location.
        float height = -(x * nx + nz * z + oo) / ny;
        // Checks for floor interaction with a 78 unit buffer.
        if (y - (height + -78.0f) < 0.0f) {
            continue;
        }

        floor_height = height;
        floor_idx = i;
        break;
    }

    if (floor_idx != -1 && floor_height > -3071.0f)
    {
        marioPos[1] = floor_height;

        mat[1][0] = normal[0];
        mat[1][1] = normal[1];
        mat[1][2] = normal[2];

        invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

        mat[1][0] *= invsqrt;
        mat[1][1] *= invsqrt;
        mat[1][2] *= invsqrt;

        mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
        mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
        mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

        invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

        mat[0][0] *= invsqrt;
        mat[0][1] *= invsqrt;
        mat[0][2] *= invsqrt;

        mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
        mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
        mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

        invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

        mat[2][0] *= invsqrt;
        mat[2][1] *= invsqrt;
        mat[2][2] *= invsqrt;

        mat[3][0] = platformPos[0];
        mat[3][1] = platformPos[1];
        mat[3][2] = platformPos[2];
        mat[0][3] = 0.0f;
        mat[1][3] = 0.0f;
        mat[2][3] = 0.0f;
        mat[3][3] = 1.0f;

        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < 3; i++) {
                float vx = defaultTriangles[h][i][0];
                float vy = defaultTriangles[h][i][1];
                float vz = defaultTriangles[h][i][2];

                currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
                currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
                currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
            }

            triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
            triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
            triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

            invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

            triangleNormals[h][0] *= invsqrt;
            triangleNormals[h][1] *= invsqrt;
            triangleNormals[h][2] *= invsqrt;

            originOffsets[h] = -(triangleNormals[h][0] * currentTriangles[h][0][0] + triangleNormals[h][1] * currentTriangles[h][0][1] + triangleNormals[h][2] * currentTriangles[h][0][2]);
        }

        bool oTiltingPyramidMarioOnPlatform = false;
        bool onPlatform = false;

        for (int i = 0; i < frames; i++) {
            float dx;
            float dy;
            float dz;
            float d;

            float dist[3];
            float posBeforeRotation[3];
            float posAfterRotation[3];

            // Mario's position
            float mx;
            float my;
            float mz;

            int marioOnPlatform = 0;

            if (onPlatform)
            {
                mx = marioPos[0];
                my = marioPos[1];
                mz = marioPos[2];

                dist[0] = mx - (float)platformPos[0];
                dist[1] = my - (float)platformPos[1];
                dist[2] = mz - (float)platformPos[2];

                for (int i = 0; i < 3; i++) {
                    posBeforeRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
                }

                dx = mx - (float)platformPos[0];
                dy = 500.0f;
                dz = mz - (float)platformPos[2];
                d = sqrtf(dx * dx + dy * dy + dz * dz);

                //! Always true since dy = 500, making d >= 500.
                if (d != 0.0f) {
                    // Normalizing
                    d = 1.0 / d;
                    dx *= d;
                    dy *= d;
                    dz *= d;
                }
                else {
                    dx = 0.0f;
                    dy = 1.0f;
                    dz = 0.0f;
                }

                if (oTiltingPyramidMarioOnPlatform == true)
                    marioOnPlatform++;
                oTiltingPyramidMarioOnPlatform = true;
            }
            else
            {
                dx = 0.0f;
                dy = 1.0f;
                dz = 0.0f;
                oTiltingPyramidMarioOnPlatform = false;
            }

            // Approach the normals by 0.01f towards the new goal, then create a transform matrix and orient the object. 
            // Outside of the other conditionals since it needs to tilt regardless of whether Mario is on.

            normal[0] = (normal[0] <= dx) ? ((dx - normal[0] < 0.01f) ? dx : (normal[0] + 0.01f)) : ((dx - normal[0] > -0.01f) ? dx : (normal[0] - 0.01f));
            normal[1] = (normal[1] <= dy) ? ((dy - normal[1] < 0.01f) ? dy : (normal[1] + 0.01f)) : ((dy - normal[1] > -0.01f) ? dy : (normal[1] - 0.01f));
            normal[2] = (normal[2] <= dz) ? ((dz - normal[2] < 0.01f) ? dz : (normal[2] + 0.01f)) : ((dz - normal[2] > -0.01f) ? dz : (normal[2] - 0.01f));

            mat[1][0] = normal[0];
            mat[1][1] = normal[1];
            mat[1][2] = normal[2];

            invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

            mat[1][0] *= invsqrt;
            mat[1][1] *= invsqrt;
            mat[1][2] *= invsqrt;

            mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
            mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
            mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

            invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

            mat[0][0] *= invsqrt;
            mat[0][1] *= invsqrt;
            mat[0][2] *= invsqrt;

            mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
            mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
            mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

            invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

            mat[2][0] *= invsqrt;
            mat[2][1] *= invsqrt;
            mat[2][2] *= invsqrt;

            mat[3][0] = platformPos[0];
            mat[3][1] = platformPos[1];
            mat[3][2] = platformPos[2];
            mat[0][3] = 0.0f;
            mat[1][3] = 0.0f;
            mat[2][3] = 0.0f;
            mat[3][3] = 1.0f;

            for (int h = 0; h < 2; h++) {
                for (int i = 0; i < 3; i++) {
                    float vx = defaultTriangles[h][i][0];
                    float vy = defaultTriangles[h][i][1];
                    float vz = defaultTriangles[h][i][2];

                    currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
                    currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
                    currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
                }

                triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
                triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
                triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

                invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

                triangleNormals[h][0] *= invsqrt;
                triangleNormals[h][1] *= invsqrt;
                triangleNormals[h][2] *= invsqrt;

                originOffsets[h] = -(triangleNormals[h][0] * currentTriangles[h][0][0] + triangleNormals[h][1] * currentTriangles[h][0][1] + triangleNormals[h][2] * currentTriangles[h][0][2]);
            }

            // If Mario is on the platform, adjust his position for the platform tilt.
            if (marioOnPlatform) {
                for (int i = 0; i < 3; i++) {
                    posAfterRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
                }

                mx += posAfterRotation[0] - posBeforeRotation[0];
                my += posAfterRotation[1] - posBeforeRotation[1];
                mz += posAfterRotation[2] - posBeforeRotation[2];
                marioPos[0] = mx;
                marioPos[1] = my;
                marioPos[2] = mz;
            }

            floor_height = 0.0;
            floor_idx = -1;

            short x = (short)(int)marioPos[0];
            short y = (short)(int)marioPos[1];
            short z = (short)(int)marioPos[2];

            for (int i = 0; i < 2; i++) {
                short x1 = currentTriangles[i][0][0];
                short z1 = currentTriangles[i][0][2];
                short x2 = currentTriangles[i][1][0];
                short z2 = currentTriangles[i][1][2];

                // Check that the point is within the triangle bounds.
                if ((z1 - z) * (x2 - x1) - (x1 - x) * (z2 - z1) < 0) {
                    continue;
                }

                // To slightly save on computation time, set this later.
                int16_t x3 = currentTriangles[i][2][0];
                int16_t z3 = currentTriangles[i][2][2];

                if ((z2 - z) * (x3 - x2) - (x2 - x) * (z3 - z2) < 0) {
                    continue;
                }
                if ((z3 - z) * (x1 - x3) - (x3 - x) * (z1 - z3) < 0) {
                    continue;
                }

                float nx = triangleNormals[i][0];
                float ny = triangleNormals[i][1];
                float nz = triangleNormals[i][2];
                float oo = -(nx * x1 + ny * currentTriangles[i][0][1] + nz * z1);

                // Find the height of the floor at a given location.
                float height = -(x * nx + nz * z + oo) / ny;
                // Checks for floor interaction with a 78 unit buffer.
                if (y - (height + -78.0f) < 0.0f) {
                    continue;
                }

                floor_height = height;
                floor_idx = i;
                break;
            }

            onPlatform = floor_idx != -1 && fabsf(marioPos[1] - floor_height) <= 4.0;

            //Check if Mario is under the lava, or too far below the platform for it to conceivably be in reach later
            if (marioPos[1] <= -3071.0f && (floor_idx != -1 || floor_height <= -3071.0f)
                || (floor_idx != -1 && marioPos[1] - floor_height < -20.0f))
            {
                break;
            }

            float testNormal[3] = {fabs(normal[0]), fabs(normal[1]), fabs(normal[2])};

            if (testNormal[0] > testNormal[1] || testNormal[2] > testNormal[1]) {
                out[idx] = true;
                break;
            } else {
                float offset = 0.01;

                float a = testNormal[0] - offset;
                float b = testNormal[2] - offset;
                float c = testNormal[2];
                float d = sqrtf(1 - testNormal[2] * testNormal[2]);
                float sign = 1;

                float v = testNormal[1] - offset;

                float sqrt1 = sqrtf(a * a + v * v);
                float sqrt2 = sqrtf(a * a + b * b + v * v);
                float sqrt3 = sqrtf(testNormal[1] * testNormal[1] + testNormal[0] * testNormal[0]);
                float sqrt4 = sqrtf(testNormal[1] * testNormal[1] + testNormal[0] * testNormal[0] + testNormal[2] * testNormal[2]);

                float result = sign * d * sqrt1 * sqrt3 * (d * sqrt2 * (sqrt1 * testNormal[0] - a * sqrt3) * sqrt4 + c * (-sqrt1 * sqrt2 * testNormal[1] * testNormal[2] + b * v * sqrt3 * sqrt4));

                if (result < 0) {
                    out[idx] = true;
                    break;
                } else {
                    c = sqrtf(1 - testNormal[0] * testNormal[0]);
                    d = testNormal[0];
                    sign = -1;
                    result = sign * d * sqrt1 * sqrt3 * (d * sqrt2 * (sqrt1 * testNormal[0] - a * sqrt3) * sqrt4 + c * (-sqrt1 * sqrt2 * testNormal[1] * testNormal[2] + b * v * sqrt3 * sqrt4));

                    if (result < 0) {
                        out[idx] = true;
                        break;
                    }
                }
            }
        }
    }
}

int main() {
    const int nThreads = 256;
    const size_t memorySize = 10000000;

    bool* devOut = nullptr;
    cudaMalloc((void**)&devOut, memorySize * sizeof(bool));
    bool* out = (bool*)std::malloc(sizeof(bool) * memorySize);

    const int nPUFrames = 4;

    const float minNX = -0.4f;
    const float maxNX = 0.0f;
    const float minNZ = 0.2f;
    const float maxNZ = 0.6f;

    const int nSamplesNX = 101;
    const int nSamplesNZ = 101;

    const float deltaNX = (nSamplesNX > 1) ? (maxNX - minNX) / (nSamplesNX - 1) : 0;
    const float deltaNZ = (nSamplesNZ > 1) ? (maxNZ - minNZ) / (nSamplesNZ - 1) : 0;

    const float deltaX = 0.25f;
    const float deltaZ = 0.25f;

    Vec3f platformPos = {-1945.0f, -3225.0f, -715.0f};

    ofstream wf("outData.bin", ios::out | ios::binary);

    char data[nSamplesNX*nSamplesNZ];
    int idx = 0;

    for (int i = 0; i < nSamplesNX; i++) {
        for (int j = 0; j < nSamplesNZ; j++) {
            float normX = minNX + i * deltaNX;
            float normZ = minNZ + j * deltaNZ;

            Vec3f startNormal = { normX, 0.9f - (0.01f*(nPUFrames+1)), normZ };
            Platform platform = Platform(platformPos[0], platformPos[1], platformPos[2], startNormal);

            bool squishTest;

            if (normX < 0) {
                squishTest = (platform.ceilings[3].normal[1] > -0.5f);
            }
            else {
                squishTest = (platform.ceilings[0].normal[1] > -0.5f);
            }

            if (squishTest) { 
                Vec3f position = { 0.0f, 0.0f, 0.0f };

                for (int i = 0; i <= nPUFrames; i++) {
                    platform.platform_logic(position);
                }

                float minX = INT16_MAX;
                float maxX = INT16_MIN;
                float minZ = INT16_MAX;
                float maxZ = INT16_MIN;

                for (int i = 0; i < platform.triangles.size(); i++) {
                    minX = fminf(fminf(fminf(minX, platform.triangles[i].vectors[0][0]), platform.triangles[i].vectors[1][0]), platform.triangles[i].vectors[2][0]);
                    maxX = fmaxf(fmaxf(fmaxf(maxX, platform.triangles[i].vectors[0][0]), platform.triangles[i].vectors[1][0]), platform.triangles[i].vectors[2][0]);
                    minZ = fminf(fminf(fminf(minZ, platform.triangles[i].vectors[0][2]), platform.triangles[i].vectors[1][2]), platform.triangles[i].vectors[2][2]);
                    maxZ = fmaxf(fmaxf(fmaxf(maxZ, platform.triangles[i].vectors[0][2]), platform.triangles[i].vectors[1][2]), platform.triangles[i].vectors[2][2]);
                }

                int nX = round((maxX - minX) / deltaX) + 1;
                int nZ = round((maxZ - minZ) / deltaZ) + 1;

                if (nX * nZ > memorySize) {
                    printf("Warning: GPU buffer too small for normal (%g, %g), skipping.\n", normX, normZ);
                    continue;
                }

                int nBlocks = (nX * nZ + nThreads - 1) / nThreads;
                
                cudaFunc<<<nBlocks, nThreads>>>(devOut, minX, deltaX, minZ, deltaZ, nX, nZ, platform.normal[0], platform.normal[1], platform.normal[2], 100);

                cudaMemcpy(out, devOut, nX * nZ * sizeof(bool), cudaMemcpyDeviceToHost);

                bool goodPositions = false;

                for (int i = 0; i < nX * nZ; i++) {
                    if (out[i]) {
                        goodPositions = true;
                        break;
                    }
                }

                if (goodPositions) {
                    data[idx] = 255;
                }
                else {
                    data[idx] = 0;
                }

            }
            else {
                data[idx] = 0;
            }

            idx++;
        }
    }

    cudaFree(devOut);
    free(out);

    wf.write(data, nSamplesNX * nSamplesNZ);
}