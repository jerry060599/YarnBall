#include "../includes/modules/StopWatch.h"
#include "../includes/modules/KittenRendering.h"

namespace Kitten {
	StopWatch::StopWatch() {
		reset();
	}

	double StopWatch::totTime() {
		return totSecs;
	}

	double StopWatch::time(const char* tag) {
		if (gpuSync) gpuFinish();

		auto time = steady_clock::now();
		double delta = duration_cast<duration<double>>(time - lastPoint).count();
		lastPoint = time;

		totSecs += delta;
		deltaTimes.push_back(delta);
		times.push_back(totSecs);
		tags.push_back(tag);
		return totSecs;
	}

	void StopWatch::reset() {
		deltaTimes.clear();
		times.clear();
		tags.clear();

		if (gpuSync) gpuFinish();
		lastPoint = steady_clock::now();
	}

	inline void sprintTime(char buff[128], double time) {
		if (time > 60 * 60 * 3)
			snprintf(buff, 128, "%.2f hr", time / 3600);
		else if (time > 60 * 15)
			snprintf(buff, 128, "%.3f min", time / 60);
		else if (time > 10)
			snprintf(buff, 128, "%.2f sec", time);
		else if (time > 2e-3)
			snprintf(buff, 128, "%.1f ms", time * 1e3);
		else if (time > 2e-6)
			snprintf(buff, 128, "%.1f us", time * 1e6);
		else
			snprintf(buff, 128, "%.1f ns", time * 1e9);
	}

	void StopWatch::printTimes() {
		for (size_t i = 0; i < times.size(); i++) {
			char buff[128];
			sprintTime(buff, times[i]);

			if (tags[i] == nullptr)
				printf("Tag_%03zd timed @ %s", i, buff);
			else
				printf("%s timed @ %s", tags[i], buff);

			if (i > 0) {
				sprintTime(buff, deltaTimes[i]);
				printf(" delta = %s", buff);
			}
			printf("\n");
		}
		printf("Total: %.2f sec", totSecs);
		if (totSecs < 10) printf(" (%.4f ms)", 1000 * totSecs);
		printf("\n");
	}
}