#include "../YarnBall.h"
#if !defined(USE_HIP)
#include <cuda.h>
#endif
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "resample.h"

using namespace std;

namespace YarnBall {
	Sim* createFromCurves(vector<vector<vec3>>& curves, vector<bool>& isCurveClosed, int numVerts) {
		Sim* sim = new Sim(numVerts);
		numVerts = 0;
		float maxLen = 0;
		float minLen = FLT_MAX;
		for (size_t i = 0; i < curves.size(); i++) {
			auto& data = curves[i];
			Vertex* verts = sim->verts + numVerts;

			// Add the vertices
			vec3 lastP = data[0];
			verts[0].pos = lastP;
			for (size_t j = 1; j < data.size(); j++) {
				vec3 p = data[j];
				verts[j].pos = p;
				float l = glm::length(p - lastP);
				maxLen = glm::max(maxLen, l);
				minLen = glm::min(minLen, l);
				lastP = p;
			}

			// Cut off the end
			verts[data.size() - 1].flags = 0;

			// Connect the ends of closed curves
			if (isCurveClosed[i]) {
				verts[0].connectionIndex = numVerts + data.size() - 1;
				verts[data.size() - 1].connectionIndex = numVerts;
			}

			numVerts += data.size();
		}
		printf("Resampled with length max %f, min %f\n", maxLen, minLen);

		return sim;
	}

	Sim* readFromBCC(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves, bool allowResample) {
		BCCHeader header;
		FILE* pFile = fopen(path.c_str(), "rb");
		if (!pFile) throw std::runtime_error("Could not open file");
		fread(&header, sizeof(header), 1, pFile);

		// Error checking
		if (header.sign[0] != 'B' || header.sign[1] != 'C' || header.sign[2] != 'C' || header.byteCount != 0x44) {
			fprintf(stderr, "Unsupported BCC file\n");
			throw std::runtime_error("Unsupported BCC file");
		}

		bool isPolyline = header.curveType[0] == 'P' && header.curveType[1] == 'L';
		if (!isPolyline && (header.curveType[0] != 'C' || header.curveType[1] != '0'))
			throw std::runtime_error("Not polyline or uniform CMR spline");
		if (header.dimensions != 3)
			throw std::runtime_error("Only curves in 3D are supported");

		vector<vector<vec3>> curves;
		vector<bool> isCurveClosed;
		int numVerts = 0;

		// Read file into memory
		for (size_t i = 0; i < header.curveCount; i++) {
			int numPoints;
			fread(&numPoints, sizeof(int), 1, pFile);
			bool isClosed = numPoints < 0;
			numPoints = abs(numPoints);

			vector<vec3> points(numPoints);
			fread(&points[0], sizeof(vec3), numPoints, pFile);

			for (auto& p : points) p = vec3(transform * vec4(p, 1));

			if (allowResample && !isPolyline)	// Resample CMR spline
				points = Resample::resampleCMR(points, 1, points.size() - 2, targetSegLen);

			// Ignore curves with less than 3 points
			if (numPoints < 3) continue;

			isCurveClosed.push_back(isClosed && !breakUpClosedCurves);
			curves.push_back(points);
			numVerts += points.size();
		}

		fclose(pFile);

		return createFromCurves(curves, isCurveClosed, numVerts);
	}

	Sim* readFromOBJ(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves, bool allowResample) {
		ifstream file(path);
		if (!file.is_open())
			throw std::runtime_error("Could not open file.");

		vector<vec3> vertices;
		vector<vector<int>> lines;

		string line;
		while (std::getline(file, line)) {
			std::istringstream iss(line);
			std::string prefix;
			iss >> prefix;

			if (prefix == "v") {
				vec3 pos;
				iss >> pos.x >> pos.y >> pos.z;
				vertices.push_back(vec3(transform * vec4(pos, 1)));
			}
			else if (prefix == "l") {
				std::vector<int> lineIndices;
				int index;
				while (iss >> index) {
					lineIndices.push_back(index);
				}
				lines.push_back(lineIndices);
			}
		}

		int numVerts = 0;
		vector<vector<vec3>> curves;
		vector<bool> isCurveClosed;
		for (auto& line : lines) {
			vector<vec3> curve;
			curve.reserve(line.size());
			bool closed = line.front() == line.back();
			for (int index : line)
				curve.push_back(vertices[index - 1]);

			if (breakUpClosedCurves && closed) {
				curve.pop_back();
				closed = false;
			}

			if (allowResample)
				curve = Resample::resampleCMR(curve, 1, curve.size() - 2, targetSegLen);

			if (curve.size() < 4) continue;
			numVerts += curve.size();
			curves.push_back(curve);
			isCurveClosed.push_back(closed);
		}

		return createFromCurves(curves, isCurveClosed, numVerts);
	}

	Sim* readFromPoly(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves, bool allowResample) {
		std::ifstream file(path);

		if (!file.is_open())
			throw std::runtime_error("Could not open file.");

		// Read in raw data
		std::string line;
		bool isPoint = true;
		vector<vec3> points;
		vector<ivec2> segs;
		while (std::getline(file, line)) {
			std::istringstream iss(line);
			if (line == "POINTS") {
				isPoint = true;
				continue;
			}
			else if (line == "POLYS") {
				isPoint = false;
				continue;
			}
			else if (line == "END")
				break;

			if (isPoint) {
				int id;
				float x, y, z;
				char colon;
				if (iss >> id >> colon >> x >> y >> z && colon == ':') {
					points.push_back(vec3(transform * vec4(x, y, z, 1)));
					if (id != points.size())
						throw std::runtime_error("Error parsing .poly. Point id mismatch.");
				}
				else throw std::runtime_error("Error parsing .poly. Point parse error.");
			}
			else {
				int id, i, j;
				char colon;
				if (iss >> id >> colon >> i >> j && colon == ':')
					segs.push_back(ivec2(i - 1, j - 1));
				else throw std::runtime_error("Error parsing .poly. Segment parse error.");
			}
		}

		file.close();

		// Build connectivity map
		unordered_multimap<int, int> segMap;
		for (auto seg : segs) {
			segMap.insert({ seg.x, seg.y });
			segMap.insert({ seg.y, seg.x });
		}

		// Build curve through flood fill
		vector<vector<vec3>> curves;
		vector<bool> isCurveClosed;
		int numVerts = 0;
		vector<bool> visited(points.size(), false);

		// Find all open curves
		for (int i = 0; i < points.size(); i++) {
			if (visited[i]) continue;
			int numCon = segMap.count(i);
			if (numCon > 2)
				throw std::runtime_error("Graphs not allowed.");
			if (numCon > 1) continue;

			vector<vec3> curve;
			int curI = i;

			// Just keep following the curve until we get to the end
			while (!visited[curI]) {
				visited[curI] = true;
				curve.push_back(points[curI]);

				// Find the next point
				auto range = segMap.equal_range(curI);
				if (range.first == range.second) break;
				for (auto it = range.first; it != range.second; it++) {
					if (!visited[it->second]) {
						curI = it->second;
						break;
					}
				}
			}

			if (curve.size() < 4) continue;
			printf("Found open curve with %zd points from %d to %d\n", curve.size(), i + 1, curI + 1);
			curve = Resample::resampleCMR(curve, 1, curve.size() - 2, targetSegLen);
			numVerts += curve.size();
			curves.push_back(curve);
			isCurveClosed.push_back(false);
		}

		// Find all closed curves
		for (int i = 0; i < points.size(); i++) {
			if (visited[i]) continue;
			int numCon = segMap.count(i);
			if (numCon != 2)
				throw std::runtime_error("Graphs not allowed.");

			vector<vec3> curve;
			int curI = i;

			// Just keep following the curve until we loop back
			while (true) {
				visited[curI] = true;
				curve.push_back(points[curI]);

				// Find the next point
				bool found = false;
				bool isEnd = false;
				auto range = segMap.equal_range(curI);
				if (range.first == range.second) break;
				for (auto it = range.first; it != range.second; it++) {
					if (!visited[it->second]) {
						curI = it->second;
						found = true;
						break;
					}
					if (it->second == i) isEnd = true;
				}

				if (!found) {
					if (!isEnd) throw std::runtime_error("Closed curve not closed.");
					break;
				}
			}

			printf("Found closed curve with %zd points from %d to %d\n", curve.size(), i + 1, curI + 1);
			if (allowResample)
				curve = Resample::resampleCMR(curve, 1, curve.size() - 2, targetSegLen);
			if (curve.size() < 4) continue;
			numVerts += curve.size();
			curves.push_back(curve);
			isCurveClosed.push_back(!breakUpClosedCurves);
		}

		return createFromCurves(curves, isCurveClosed, numVerts);
	}
}