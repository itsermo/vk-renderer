# vk-renderer
A vulkan renderer implemented from the tutorial at https://vulkan-tutorial.com/

## Linux build instructions

0. Install prerequisite libraries:
  ```
  sudo dnf install @development-tools vulkan-devel libshaderc-devel glfw-devel
  ```

1. Install git LFS and clone the repo:
  ```
  sudo dnf install git-lfs
  git clone https://github.com/itsermo/vk-renderer
  cd vk-renderer
  ```
  
2. Build using makefile:
  ```
  make
  ```
  
3. (Optional) You can develop by opening the workspace file in "vscode" folder using Visual Studio Code.  From here you can attach debugger or build using the Visual Studio Code build tasks
