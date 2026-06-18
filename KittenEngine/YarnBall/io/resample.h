#pragma once
#include "../YarnBall.h"
#if !defined(USE_HIP)
#include <cuda.h>
#endif
#include <vector>

namespace Resample {
	using namespace std;
	using namespace glm;

	inline dvec3 sampleCurve(const vector<vec3>& cmr, double t) {
		int si = glm::clamp((int)floor(t), 1, (int)cmr.size() - 3);
		dvec3 p0 = cmr[si - 1];
		dvec3 p1 = cmr[si];
		dvec3 p2 = cmr[si + 1];
		dvec3 p3 = cmr[si + 2];

		return Kit::cmrSpline(p0, p1, p2, p3, t - si);
	}

	inline vector<double> resampleCMRCoords(const vector<vec3>& cmr, double start, double end, const double tarSegLen) {
		vector<double> ts;
		constexpr double MIN_STEP = 0.2;
		const double step = end > start ? MIN_STEP : -MIN_STEP;
		const double r2 = tarSegLen * tarSegLen;

		double i = start;
		dvec3 lastPos = sampleCurve(cmr, i);		// Last position added to the list
		ts.push_back(i);

		while ((end > start) ? i < end : i > end) {
			i += step;
			auto pos = sampleCurve(cmr, i);
			double l0 = length2(pos - lastPos);

			// We overshot the target length
			if (l0 > r2) {
				// Figure out exactly where to place this point.
				l0 = sqrt(l0);
				dvec2 range(i - step, i);
				if (range.x > range.y) std::swap(range.x, range.y);

				const double t0 = i;
				i -= 0.5 * step;	// Take half a step back
				for (int k = 0; k < 8; k++) {
					dvec3 x = sampleCurve(cmr, i);
					double d = length(x - lastPos);
					i += (i - t0) * (tarSegLen - d) / (d - l0);
				}

				i = glm::clamp(i, range.x, range.y);

				ts.push_back(i);
				lastPos = sampleCurve(cmr, i);
			}
		}

		if (ts.size() == 1)
			ts.push_back(end);
		else {
			// Move points around to make the last point line up with the end

			double d = length(sampleCurve(cmr, ts.back()) - sampleCurve(cmr, end));
			// This is the approximate segLen in coordinate space
			double dt = ts.back() - ts[ts.size() - 2];

			// Check how far the last point is from the end
			// Add a new point if its easier to do so
			if (d > 0.5f * tarSegLen)
				ts.push_back(end);
			else
				ts.back() = end;

			// Subtract it with the final one so we can correct it back to the original.
			dt -= ts.back() - ts[ts.size() - 2];

			// Now we just shift points back while making sure that the final dt is the original one.
			const int N = std::min((int)ts.size(), 8);
			for (int i = 0; i < N; i++)
				ts[ts.size() - 2 - i] -= dt * (N - i) / (double)N;
		}

		return ts;
	}

	inline vector<double> twoDirectionResampleCMRCoords(const vector<vec3>& cmr, double start, double end, const double tarSegLen) {
		// For long curves, its better to sample from both ends and then try to merge them at some point.
		vector<double> forward, backward;
#pragma omp parallel sections
		{
#pragma omp section
			{
				forward = resampleCMRCoords(cmr, start, end, tarSegLen);
			}
#pragma omp section
			{
				backward = resampleCMRCoords(cmr, end, start, tarSegLen);
			}
		}

		constexpr int BORDER = 8;
		if (forward.size() > 4 * BORDER && backward.size() > 4 * BORDER) {
			// Merge the two lists at the minimum error point
			double minError = std::numeric_limits<double>::infinity();
			ivec2 splitIndex(2 * BORDER);

			for (int i = BORDER, j = backward.size() - 1 - BORDER; i < forward.size() - 8; i++) {
				auto t = forward[i];
				while (j > 0 && backward[j] < t)
					j--;

				auto t0 = backward[j + 1];
				if (minError > abs(t - t0)) {
					minError = abs(t - t0);
					splitIndex = ivec2(i, j + 1);
				}

				auto t1 = backward[j];
				if (minError > abs(t1 - t)) {
					minError = abs(t1 - t);
					splitIndex = ivec2(i, j);
				}
			}

			// Merge the two lists
			vector<double> merged;
			merged.reserve(splitIndex.x + splitIndex.y + 1);
			merged.insert(merged.end(), forward.begin(), forward.begin() + splitIndex.x);
			double mid = 0.5 * (forward[splitIndex.x] + backward[splitIndex.y]);
			merged.push_back(mid);
			for (int i = splitIndex.y - 1; i >= 0; i--)
				merged.push_back(backward[i]);

			// Blend the split point
			double dt = forward[splitIndex.x] - mid;
			for (int i = 0; i < BORDER; i++)
				merged[splitIndex.x - i - 1] -= dt * (BORDER - i) / (1. + BORDER);
			dt = backward[splitIndex.y] - mid;
			for (int i = 0; i < BORDER; i++)
				merged[splitIndex.x + i + 1] -= dt * (BORDER - i) / (1. + BORDER);

			return merged;
		}
		else return forward;
	}

	// Simple algorithm to resample a CMR spline
	inline vector<vec3> resampleCMR(const vector<vec3>& cmr, double start, double end, double tarSegLen) {
		// We resample the curve twice
		// The first time gets us average arc length
		// The second time we resample with the average arc length
		// Works best for short curves
		// On long curves we hope the first pass is good enough

		// Get total seglen
		auto ts0 = twoDirectionResampleCMRCoords(cmr, start, end, tarSegLen);
		double totalSegLen = 0;
		double error0 = 0;
		auto lastPos = sampleCurve(cmr, ts0[0]);
		for (size_t i = 1; i < ts0.size(); i++) {
			auto pos = sampleCurve(cmr, ts0[i]);
			double len = length(pos - lastPos);
			error0 = std::max(error0, glm::abs(len - tarSegLen));
			totalSegLen += len;
			lastPos = pos;
		}

		// Resample again with average segment length
		const double avgSegLen = totalSegLen / (ts0.size() - 1);
		auto ts1 = twoDirectionResampleCMRCoords(cmr, start, end, avgSegLen);
		lastPos = sampleCurve(cmr, ts1[0]);
		double error1 = 0;
		for (size_t i = 1; i < ts1.size(); i++) {
			auto pos = sampleCurve(cmr, ts1[i]);
			double len = length(pos - lastPos);
			error1 = std::max(error1, glm::abs(len - avgSegLen));
			lastPos = pos;
		}

		// Sample from whichever has the lowest error
		auto& ts = error0 < error1 ? ts0 : ts1;

		vector<vec3> resampled(ts.size());
#pragma omp parallel for schedule(static, 1024)
		for (int i = 0; i < ts.size(); i++)
			resampled[i] = (vec3)sampleCurve(cmr, ts[i]);

		return resampled;
	}
}