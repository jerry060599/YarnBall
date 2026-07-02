# YarnBall Sim: A high performance yarn simulator
 [![](https://www.replicabilitystamp.org/logo/Reproducibility-small.png)](http://www.replicabilitystamp.org#https-github-com-jerry060599-yarnball)
 
A massively parallel GPU implementation of the paper ["Stable Cosserat Rods" SIGGRAPH 2025](https://jerryhsu.io/projects/StableCosseratRods/), [Jerry Hsu](https://jerryhsu.io), Tongtong Wang, Kui Wu, and Cem Yuksel. 

This repository contains the source code, model, and paramters for the yarn twisting and yarn letter examples.
For the CPU based examples, see the [StableCosseratRods](https://github.com/jerry060599/StableCosseratRods) repo.

[![Example](images/yarnTwist.gif)](https://youtu.be/hmEGLPG1zP0)

^ The above example runs in real time on an NVidia RTX 3090. 

## Build
Project templated from [Kitten Engine](https://github.com/jerry060599/KittenEngine/tree/main)

Configured for Windows and Visual Studios. 

**Requires/tested on CUDA 12.8**

**Dependencies using vcpkg**: assimp, eigen3, stb headers, glad, glfw, imgui[opengl3-glad-binding], glm, jsoncpp, cli11

To install these packages:

1. Setup vcpkg https://vcpkg.io/en/getting-started.html
```
git clone https://github.com/microsoft/vcpkg.git
.\bootstrap-vcpkg.bat
.\vcpkg.exe integrate install
```
Optionally add vcpkg.exe to %PATH%

2. Run:
```
vcpkg.exe install glm:x64-windows
vcpkg.exe install eigen3:x64-windows
vcpkg.exe install assimp:x64-windows
vcpkg.exe install glad:x64-windows
vcpkg.exe install freetype:x64-windows
vcpkg.exe install jsoncpp:x64-windows
vcpkg.exe install imgui[core,glfw-binding,opengl3-binding]:x64-windows
vcpkg.exe install cli11:x64-windows
```

**DO NOT INSTALL GLM DIRECTLY FROM THE CURRENT WORKING BRANCH.**
Only install versions taged as stable releases or through vcpkg. 
When in doubt, use glm version ```1.0.1#3```. 

This was tested to run on the NVIDIA RTX 3090, 4090, and 5090. 
It was also tested with the RTX 2080 by changing the CUDA code generation setting to use "compute_75,sm_75". 

### Building on Linux with CMake (NVIDIA CUDA or AMD ROCm)

A `CMakeLists.txt` at the repo root builds the simulator on Linux for either NVIDIA CUDA (default) or AMD GPUs via ROCm/HIP (`-DUSE_HIP=ON`). GLM 1.0.1 and Dear ImGui are fetched automatically; the remaining dependencies (glfw, OpenGL, Freetype, assimp, CLI11, Eigen3, jsoncpp, stb) come from the system package manager. Run `git lfs pull` first to fetch the `.bcc` model files.

NVIDIA CUDA:
```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

AMD ROCm/HIP (set the gfx target for your GPU, e.g. `gfx90a`, `gfx1100`, `gfx1201`):
```bash
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The headless simulation path (`--headless`) runs the full GPU compute pipeline without windowing; the interactive GUI and its GL/GPU interop currently target the Windows/Visual Studio build.

## Usage
This repo contains both a sample CLI and a C++ interface. 
In most cases, it is the easiest to load a scene directly through a provided JSON.
Sample JSON formats can be found in [cable_work_pattern.json](KittenEngine/configs/cable_work_pattern.json) and [letterS.json](KittenEngine/configs/letterS.json).
Curves/lines can be provided with [.bcc](https://www.cemyuksel.com/cyCodeBase/soln/using_bcc_files.html), [.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file), or [.poly](https://paulbourke.net/dataformats/poly/) files. See [reader.cpp](KittenEngine/YarnBall/io/reader.cpp) for how to load additional formats.

### CLI
The CLI contains some basic functionality for simulating and exporting scenes.

![Gui](images/gui.gif)

By default, the CLI launches a GUI for visualization and additional controls. Left click to move. Right click to pan. Scroll to zoom. Space to pause/unpause. Additionally, the GUI can be turned off with the ```--headless``` flag. 

Run ``` Gui.exe -help``` to see other availible flags.
```
YarnBall: High performance Cosserat Rods simulation.
Usage: C:\Workspace\Yarn\YarnBall\x64\Release\Gui.exe [OPTIONS] [filename]

Positionals:
  filename TEXT               Path to the scene json file

Options:
  -h,--help                   Print this help message and exit
  -o,--output TEXT            Output path prefix (directory must exist). Output file path if last frame only.
  -n,--nframes INT            Number of frames to simulate
  --headless                  Run in headless mode (without GUI)
  --exit                      Exit once all exports are done.
  -s                          Start simulating immediately
  -e,--export                 Export simulation frames
  --exportlast                Export the last frame only
  --fiber                     Export as fiber level mesh (slow) instead of obj splines
  --bcc                       Export as BCC format instead of obj splines
  --twist                     Twist animation
  --pull                      Pull animation
  --fps INT [30]              Animation frames per second
```

For example, to run the twisting example in the paper, use
```
Gui.exe configs\cable_work_pattern.json --twist -s
```
To export the animation, use
```
Gui.exe configs\cable_work_pattern.json --export --twist -s -n 750
```

### C++ interface
The C++ interface includes everything under the YarnBall namespace and is placed under its separate VS project, "YarnBall".
See [YarnBall.h](KittenEngine/YarnBall/YarnBall.h) for availible functions and [jsonBuilder.cpp](KittenEngine/YarnBall/io/jsonBuilder.cpp) for sample usage.

To use the simulator, first create a ```YarnBall::Sim``` object with the predetermined number of vertices. 
Once created, vertex positions can be populated within the ```sim->verts``` array. 

To create static vertices, simply set ```sim->verts[i].invMass = 0```. In YarnBall, vertex_i is automatically connected with verteix_{i+1} to form segments.
To create breaks into multiple distinct yarn pieces, unset ```VertexFlags::hasNext``` in the bit flag ```sim->verts[i].flags```. 
This can be accomplished by simply setting the entire flag to 0. The rest of the bit flags are automatically populated after ```configure()```. 

Once done, call ```configure()```. ```configure()``` will automatically populate the proper flags, segment orientations, and rest shapes. 
Simulation can then be done by calling ```sim->advance(1/30.f)``` with the desired frame time. 

Note: The data is passed off to the GPU after ```configure()```. As such, any simulation results will need to call ```download()``` for them to be reflected in ```sim->verts```.
Conversely, any changes to ```sim->verts``` needs to call ```upload()``` to take effect. 

For example, the following code creates and simulates 5 seconds of two small pieces of stiff curved yarn. 
```
constexpr int numVerts = 64;
auto sim = new YarnBall::Sim(numVerts);
const float segLen = 0.002f;

for (int i = 0; i < 32; i++) {
	vec3 pos = vec3(0.00002f * i * (i - 16) + segLen * 1, -segLen * i, 0);
	sim->verts[i].pos = pos;
	pos.x *= -1;
	sim->verts[i + 32].pos = pos;
}

sim->verts[0].invMass = sim->verts[32].invMass = sim->verts[63].invMass = 0;
sim->verts[31].flags = 0;
sim->meta.kCollision = 1e-7;
sim->configure();
sim->setKBend(1e-8);
sim->setKStretch(2e-2);
sim->maxH = 1e-3;
sim->upload();

sim->advance(5.f);
```
See [jsonBuilder.cpp](KittenEngine/YarnBall/io/jsonBuilder.cpp) for more availible controls and settings. 

## License
This software is licensed under the GPLv3 License.
Unless otherwise stated in the file header, the contents of this repository are provided under the following license. Files that specify a different copyright are governed by the terms indicated therein.

Copyright (C) 2025 Jerry Hsu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

For commercial use beyond the existing license, you should hire me at jerry.hsu.research@gmail.com :) 
